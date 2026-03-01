"""
Coral experiment runner — DEVELOPMENT VERSION.

Active development branch. Diverges from the thesis version (experiments/coral/)
with physics fixes and ongoing improvements. Current changes vs. thesis:

  - softmax invest/liquidate → tanh signed trade (physics.py activate_outputs)

See logs/ for design rationale on each change.

Usage:
    python experiments/coral_dev/run.py
    python experiments/coral_dev/run.py --no-gui --steps 300 --profile
    python experiments/coral_dev/run.py --benchmark --steps 200 --shape 400
    python experiments/coral_dev/run.py --env hole --env-param 0.35
    python experiments/coral_dev/run.py --env stripes --env-param 6
    python experiments/coral_dev/run.py --backend cpu --device cpu --no-gui --steps 100
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
parser = argparse.ArgumentParser(description="Coral Dev Experiment Runner")
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--shape", type=int, default=400)
parser.add_argument("--backend", type=str, default="metal",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--radiate-interval", type=int, default=50)
parser.add_argument("--cull-max-pop", type=int, default=100)
parser.add_argument("--checkpoint-interval", type=int, default=1000,
                    help="Steps between checkpoints (0 = disable)")
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--env", type=str, default="flat",
                    choices=["flat", "hole", "ring", "stripes", "corners"])
parser.add_argument("--env-param", type=float, default=None)
parser.add_argument("--resume-from", type=str, default=None,
                    help="Path to a checkpoint dir; resume run from that state. "
                         "--steps counts additional steps from the checkpoint.")
args = parser.parse_args()

import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

backend_map = {"cpu": ti.cpu, "metal": ti.metal,
               "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.evolver import apply_weights_and_biases
from coralai.visualization import Visualization
from experiment import EXPERIMENT as exp


# ---------------------------------------------------------------------------
# GUI overlay
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
    import json, shutil

    shape = (args.shape, args.shape)
    substrate = exp.make_substrate(shape, DEVICE)

    if args.benchmark:
        inds = substrate.ti_indices[None]
        substrate.mem[0, inds.energy] = torch.rand(shape, device=DEVICE)
        substrate.mem[0, inds.infra]  = torch.rand(shape, device=DEVICE)
        env = exp.make_env("flat")
    else:
        env = exp.make_env(args.env, args.env_param)
        env.seed(substrate)

    evolver = exp.make_evolver(substrate)

    # Load checkpoint if resuming; otherwise this is step 0
    if args.resume_from:
        evolver.load_checkpoint(args.resume_from)
        print(f"  [resume] loaded {args.resume_from}  (step {evolver.timestep})")
    start_step = evolver.timestep

    headless = args.no_gui or args.benchmark
    vis = None if headless else exp.make_vis(substrate, evolver)

    inds = substrate.ti_indices[None]
    timings = defaultdict(float)
    step_times = []
    fps_window = []
    fps_print_interval = 2.0
    last_fps_time = time.time()
    last_fps = 0.0
    t_session = time.time()
    # end_step is absolute; --steps counts additional steps from wherever we start
    end_step = start_step + (args.steps if args.steps > 0 else 10 ** 9)

    # ---- Run directory ---------------------------------------------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    run_dir = os.path.join(repo_root, "runs",
                           f"coral_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    shutil.copytree(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(run_dir, "snapshot"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "experiment":   exp.name,
            "shape":        list(shape),
            "seed":         args.seed,
            "env":          env.to_dict(),
            "env_persist":  env.persist,
            "start_step":   start_step,
            "resumed_from": args.resume_from,
            "start_time":   datetime.now().isoformat(),
        }, f, indent=2)

    if not args.benchmark:
        torch.save(substrate.mem.cpu(), os.path.join(run_dir, "initial_state.pt"))

    log_path = os.path.join(run_dir, "step_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "total_energy", "total_infra", "energy_pct",
                         "n_alive", "n_genomes", "energy_offset", "fps"])
    # ---------------------------------------------------------------------------

    print(f"Run dir: {run_dir}")
    print(f"Coral-dev | {shape[0]}x{shape[1]} | {args.backend}/{args.device} | "
          f"env={args.env} persist={env.persist} | GUI={'OFF' if headless else 'ON'} | "
          f"{'benchmark' if args.benchmark else 'profile' if args.profile else 'run'}")
    if start_step > 0:
        print(f"  Resuming from step {start_step}")
    print("-" * 60)

    # Save step-0 checkpoint on fresh runs so replay can always start from the beginning
    if not args.benchmark and args.checkpoint_interval > 0 and start_step == 0:
        evolver.save_checkpoint(run_dir, 0)

    from evolution import apply_radiation_mutation

    for step in range(start_step, end_step):
        t_step = time.perf_counter()

        sync(); t0 = time.perf_counter()
        cw, cb = evolver.get_combined_weights()
        out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                 cw, cb, evolver.dir_kernel, evolver.dir_order,
                                 substrate.ti_indices)
        substrate.mem[0, evolver.act_chinds] = out_mem
        sync(); timings["0_nn_forward"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        exp.run_physics(substrate, evolver)
        sync(); timings["1_physics"] += time.perf_counter() - t0

        sync(); t0 = time.perf_counter()
        exp.run_evolution(substrate, evolver, step)
        sync(); timings["2_evolution"] += time.perf_counter() - t0

        if env.persist:
            env.step(substrate)

        if vis:
            sync(); t0 = time.perf_counter()
            vis.update()
            sync(); timings["3_render"] += time.perf_counter() - t0

        if step % args.radiate_interval == 0 and step > 0:
            sync(); t0 = time.perf_counter()
            apply_radiation_mutation(evolver, 5)
            sync(); timings["4_radiation"] += time.perf_counter() - t0

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
        print("\n  Per-category breakdown:")
        for name, t in sorted(timings.items()):
            pct = 100 * t / total_p if total_p > 0 else 0
            print(f"    {name:25s}  {1000*t/n:6.2f}ms/step  {pct:5.1f}%")
        print(f"    {'TOTAL':25s}  {1000*total_p/n:6.2f}ms/step")


if __name__ == "__main__":
    main()
