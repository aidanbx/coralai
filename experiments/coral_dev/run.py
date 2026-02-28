"""
Coral experiment runner — DEVELOPMENT VERSION.

Active development branch. Diverges from the thesis version (experiments/coral/)
with physics fixes and ongoing improvements. Current changes vs. thesis:

  - softmax invest/liquidate → tanh signed trade (physics.py activate_outputs)

See logs/ for design rationale on each change.

Runs the spatial NEAT coral simulation with optional GUI, profiling, or
benchmark modes.

Usage:
    # Interactive GUI (default):
    python experiments/coral/run.py

    # Headless for N steps with per-function profiling:
    python experiments/coral/run.py --no-gui --steps 300 --profile

    # Benchmark (worst-case, all cells alive, seeded RNG):
    python experiments/coral/run.py --benchmark --steps 200 --shape 400

    # Smaller grid for quick experimentation:
    python experiments/coral/run.py --shape 200 --steps 10000

    # CPU-only (headless Linux, no Metal):
    python experiments/coral/run.py --backend cpu --device cpu --no-gui --steps 100
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import torch
import taichi as ti

# ---------------------------------------------------------------------------
# Argument parsing (before ti.init so --help works without GPU init)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Coral Space Experiment Runner")
parser.add_argument("--steps", type=int, default=0,
                    help="Steps to run then exit (0 = run until window closed)")
parser.add_argument("--shape", type=int, default=400,
                    help="Grid side length (shape x shape)")
parser.add_argument("--backend", type=str, default="metal",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true",
                    help="Headless: skip rendering")
parser.add_argument("--profile", action="store_true",
                    help="Print per-function timing breakdown on exit")
parser.add_argument("--benchmark", action="store_true",
                    help="Seed RNG, fill all cells, measure worst-case throughput")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (used with --benchmark)")
parser.add_argument("--radiate-interval", type=int, default=50)
parser.add_argument("--cull-max-pop", type=int, default=100)
parser.add_argument("--checkpoint-interval", type=int, default=1000,
                    help="Steps between checkpoints (0 = disable)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="Steps between CSV log rows")
args = parser.parse_args()

backend_map = {"cpu": ti.cpu, "metal": ti.metal,
               "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.substrate import Substrate
from coralai.evolver import SpaceEvolver, apply_weights_and_biases
from coralai.visualization import Visualization

# ---------------------------------------------------------------------------
# Channel / kernel config (the experiment definition)
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy": ti.f32,
    "infra": ti.f32,
    "acts": ti.types.struct(
        invest=ti.f32,
        liquidate=ti.f32,
        explore=ti.types.vector(n=4, dtype=ti.f32),  # no, fwd, left, right
    ),
    "com": ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
    "rot": ti.f32,
    "genome": ti.f32,
}
KERNEL = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
          [-1, 0], [-1, -1], [0, -1], [1, -1]]
DIR_ORDER = [0, -1, 1]
SENSE_CHS = ["energy", "infra", "com"]
ACT_CHS = ["acts", "com"]

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neat.config")

# ---------------------------------------------------------------------------
# GUI overlay (only used when not --no-gui)
# ---------------------------------------------------------------------------
class CoralVis(Visualization):
    def __init__(self, substrate, evolver, vis_chs):
        super().__init__(substrate, vis_chs, panel_width=300)
        self.evolver = evolver
        self.genome_stats = []
        self.fps_history = []

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        px = self.panel_x
        pw = self.panel_wfrac
        n_chs = self.substrate.mem.shape[1]

        # -- Panel 1: Display — y=0.01..0.46 ---------------------------------
        with self.gui.sub_window("Display", px, 0.01, pw, 0.46) as sw:
            self._draw_norm_controls(sw)
            if sw.button("E | I | Rot"):
                self.chinds[0] = int(inds.energy)
                self.chinds[1] = int(inds.infra)
                self.chinds[2] = int(inds.rot)
                self.view_mode = 0
            if sw.button("Genome | E | I"):
                self.chinds[0] = int(inds.genome)
                self.chinds[1] = int(inds.energy)
                self.chinds[2] = int(inds.infra)
                self.view_mode = 0
            if sw.button("Energy only"):
                self.chinds[0] = int(inds.energy)
                self.view_mode = 1
            if sw.button("Infra only"):
                self.chinds[1] = int(inds.infra)
                self.view_mode = 2
            if sw.button("Genome only"):
                self.chinds[0] = int(inds.genome)
                self.view_mode = 1
            self.chinds[0] = sw.slider_int(
                f"R: {self.substrate.index_to_chname(int(self.chinds[0]))}",
                int(self.chinds[0]), 0, n_chs - 1)
            self.chinds[1] = sw.slider_int(
                f"G: {self.substrate.index_to_chname(int(self.chinds[1]))}",
                int(self.chinds[1]), 0, n_chs - 1)
            self.chinds[2] = sw.slider_int(
                f"B: {self.substrate.index_to_chname(int(self.chinds[2]))}",
                int(self.chinds[2]), 0, n_chs - 1)
            self.paused = sw.checkbox("Pause", self.paused)

        # -- Panel 2: Stats — y=0.47..0.99 -----------------------------------
        with self.gui.sub_window("Stats", px, 0.47, pw, 0.52) as sw:
            pos = self.window.get_cursor_pos()
            sim_frac = self.sim_w / self.image.shape[0]
            cx = int((pos[0] / sim_frac) * self.w) % self.w
            cy = int(pos[1] * self.h) % self.h
            sw.text(
                f"GENOME: {self.substrate.mem[0, inds.genome, cx, cy]:.0f}\n"
                f"Energy: {self.substrate.mem[0, inds.energy, cx, cy]:.2f}\n"
                f"Infra:  {self.substrate.mem[0, inds.infra, cx, cy]:.2f}")
            sw.text(f"Step: {self.evolver.timestep}")
            tot_e = torch.sum(self.substrate.mem[0, inds.energy])
            tot_i = torch.sum(self.substrate.mem[0, inds.infra])
            sw.text(f"Total E+I: {tot_e + tot_i:.1f}")
            sw.text(f"Energy%: {100 * tot_e / (tot_e + tot_i):.1f}%")
            sw.text(f"E offset: {self.evolver.energy_offset:.3f}")
            sw.text(f"Genomes: {len(self.evolver.genomes)}")
            if self.fps_history:
                sw.text(f"FPS: {self.fps_history[-1]:.1f}")
            if self.evolver.timestep % 20 == 0:
                self.genome_stats = []
                for i in range(len(self.evolver.genomes)):
                    n = self.substrate.mem[0, inds.genome].eq(i).sum().item()
                    self.genome_stats.append((i, n, self.evolver.ages[i]))
                self.genome_stats.sort(key=lambda x: x[1], reverse=True)
            for i, n_cells, age in self.genome_stats[:8]:
                sw.text(f"  G{i}: {n_cells:.0f}c  {age}t")


# ---------------------------------------------------------------------------
# Device sync helper
# ---------------------------------------------------------------------------
def sync():
    if args.device == "mps":
        torch.mps.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    shape = (args.shape, args.shape)

    if args.benchmark:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    substrate = Substrate(shape, torch.float32, DEVICE, CHANNELS)
    substrate.malloc()

    if args.benchmark:
        # Fill all cells to benchmark worst-case (maximum computation)
        inds = substrate.ti_indices[None]
        substrate.mem[0, inds.energy] = torch.rand(shape, device=DEVICE)
        substrate.mem[0, inds.infra] = torch.rand(shape, device=DEVICE)

    evolver = SpaceEvolver(CONFIG_PATH, substrate, KERNEL, DIR_ORDER,
                           SENSE_CHS, ACT_CHS)

    headless = args.no_gui or args.benchmark
    vis = None if headless else CoralVis(substrate, evolver,
                                         ["energy", "infra", "rot"])

    inds = substrate.ti_indices[None]
    timings = defaultdict(float)
    step_times = []
    fps_window = []
    fps_print_interval = 2.0
    last_fps_time = time.time()
    last_fps = 0.0
    t_session = time.time()
    max_steps = args.steps if args.steps > 0 else 10 ** 9

    # ---- Run directory + CSV logger ----------------------------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    run_dir = os.path.join(
        repo_root, "runs",
        f"coral_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "step_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "total_energy", "total_infra", "energy_pct",
        "n_alive", "n_genomes", "energy_offset", "fps",
    ])
    print(f"Run dir: {run_dir}")
    # ------------------------------------------------------------------------

    print(f"Coral-dev | {shape[0]}x{shape[1]} | {args.backend}/{args.device} | "
          f"GUI={'OFF' if headless else 'ON'} | "
          f"{'benchmark' if args.benchmark else 'profile' if args.profile else 'run'}")
    print("-" * 60)

    from physics import activate_outputs, invest_liquidate, explore_physics, energy_physics

    for step in range(max_steps):
        t_step = time.perf_counter()

        sync(); t0 = time.perf_counter()
        cw, cb = evolver.get_combined_weights()
        sync(); timings["0_get_weights"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                 cw, cb, evolver.dir_kernel, evolver.dir_order,
                                 substrate.ti_indices)
        substrate.mem[0, evolver.act_chinds] = out_mem
        sync(); timings["1_apply_weights"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        activate_outputs(substrate)
        sync(); timings["2_activate_out"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        invest_liquidate(substrate)
        sync(); timings["3_invest_liq"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        explore_physics(substrate, evolver.kernel, evolver.dir_order)
        sync(); timings["4_explore"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        energy_physics(substrate, evolver.kernel, max_infra=10, max_energy=1.5)
        sync(); timings["5_energy"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        alive = (substrate.mem[0, inds.infra] + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)
        sync(); timings["6_death"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        evolver.energy_offset = evolver.get_energy_offset(step)
        evolver.ages = [a + 1 for a in evolver.ages]
        offset = evolver.energy_offset
        substrate.mem[0, inds.energy].add_(
            torch.randn_like(substrate.mem[0, inds.energy]).add_(offset).mul_(0.1))
        substrate.mem[0, inds.infra].add_(
            torch.randn_like(substrate.mem[0, inds.infra]).add_(offset).mul_(0.1))
        substrate.mem[0, inds.energy].clamp_(0.01, 100)
        substrate.mem[0, inds.infra].clamp_(0.01, 100)
        if step % 50 == 0:
            evolver.kill_random_chunk(5)
        sync(); timings["7_step_rest"] += time.perf_counter() - t0

        if vis:
            sync(); t0 = time.perf_counter()
            vis.update()
            sync(); timings["8_render"] += time.perf_counter() - t0

        if step % args.radiate_interval == 0 and step > 0:
            sync(); t0 = time.perf_counter()
            evolver.apply_radiation_mutation(5)
            sync(); timings["9_radiation"] += time.perf_counter() - t0

        if (len(evolver.genomes) > args.cull_max_pop
                and (step - evolver.time_last_cull) > 50):
            evolver.reduce_population_to_threshold(args.cull_max_pop)

        evolver.timestep = step + 1
        dt = time.perf_counter() - t_step
        step_times.append(dt)

        now = time.time()
        fps_window.append(dt)
        if now - last_fps_time >= fps_print_interval:
            last_fps = len(fps_window) / sum(fps_window)
            alive_n = (substrate.mem[0, inds.genome] >= 0).sum().item()
            if vis:
                vis.fps_history.append(last_fps)
            print(f"  step {step+1:5d} | {last_fps:5.1f} FPS | "
                  f"{1000*sum(fps_window)/len(fps_window):5.1f}ms/step | "
                  f"alive: {alive_n:,} | genomes: {len(evolver.genomes)}")
            fps_window = []
            last_fps_time = now

        # ---- CSV logging ----------------------------------------------------
        if not args.benchmark and step % args.log_interval == 0:
            tot_e = float(substrate.mem[0, inds.energy].sum())
            tot_i = float(substrate.mem[0, inds.infra].sum())
            alive_n = int((substrate.mem[0, inds.genome] >= 0).sum())
            log_writer.writerow([
                step, f"{tot_e:.4f}", f"{tot_i:.4f}",
                f"{100 * tot_e / (tot_e + tot_i + 1e-8):.2f}",
                alive_n, len(evolver.genomes),
                f"{evolver.energy_offset:.4f}", f"{last_fps:.2f}",
            ])
            log_file.flush()

        # ---- Periodic checkpoint --------------------------------------------
        if (not args.benchmark and args.checkpoint_interval > 0
                and step > 0 and step % args.checkpoint_interval == 0):
            evolver.save_checkpoint(run_dir, step)

        if vis and not vis.window.running:
            break

    log_file.close()

    total = time.time() - t_session
    n = len(step_times)
    avg_fps = n / sum(step_times)
    median_ms = sorted(step_times)[n // 2] * 1000
    alive_final = (substrate.mem[0, inds.genome] >= 0).sum().item()

    print(f"\n{'='*60}")
    print(f"  {n} steps in {total:.2f}s  |  avg {avg_fps:.1f} FPS  |  "
          f"median {median_ms:.1f}ms/step")
    print(f"  alive: {alive_final:,}  |  genomes: {len(evolver.genomes)}")
    print(f"  log: {log_path}")
    print(f"{'='*60}")

    if args.profile or args.benchmark:
        total_p = sum(timings.values())
        print("\n  Per-function breakdown:")
        for name, t in sorted(timings.items()):
            pct = 100 * t / total_p if total_p > 0 else 0
            print(f"    {name:25s}  {1000*t/n:6.2f}ms/step  {pct:5.1f}%")
        print(f"    {'TOTAL':25s}  {1000*total_p/n:6.2f}ms/step")


if __name__ == "__main__":
    main()
