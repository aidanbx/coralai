"""
coral_dev replay — load a past run and navigate between saved checkpoints.

Loads a runs/coral_dev_TIMESTAMP/ directory, discovers all checkpoints, and
lets you scrub between them visually. With the RNG state restored from each
checkpoint, running forward from any point reproduces the original trajectory
(bitwise-identical on CPU; near-identical on MPS due to parallel-reduction FP).

Usage:
    # Open GUI at latest checkpoint of a run:
    python experiments/coral_dev/replay.py --run-dir runs/coral_dev_20260228_174936

    # Start at a specific checkpoint step:
    python experiments/coral_dev/replay.py --run-dir runs/coral_dev_... --step 2000

    # CPU backend for bitwise-identical replay:
    python experiments/coral_dev/replay.py --run-dir runs/coral_dev_... \\
        --backend cpu --device cpu
"""

import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict

import torch
import taichi as ti

# ---------------------------------------------------------------------------
# Argument parsing — before ti.init so --help works without GPU init
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="coral_dev checkpoint replay")
parser.add_argument("--run-dir", type=str, required=True,
                    help="Path to runs/coral_dev_TIMESTAMP/ directory")
parser.add_argument("--step", type=int, default=None,
                    help="Checkpoint step to load (default: latest available)")
parser.add_argument("--backend", type=str, default="metal",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "mps", "cuda"])
args = parser.parse_args()

backend_map = {"cpu": ti.cpu, "metal": ti.metal,
               "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(backend_map[args.backend])
DEVICE = torch.device(args.device)

from coralai.substrate import Substrate
from coralai.evolver import SpaceEvolver, apply_weights_and_biases
from coralai.visualization import Visualization

# ---------------------------------------------------------------------------
# Channel / kernel config — must match the run that generated the checkpoint
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy": ti.f32,
    "infra": ti.f32,
    "acts": ti.types.struct(
        invest=ti.f32,
        liquidate=ti.f32,
        explore=ti.types.vector(n=4, dtype=ti.f32),
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

# neat.config is taken from the snapshot so the exact NEAT params are preserved
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neat.config")


# ---------------------------------------------------------------------------
# Load physics + evolution from snapshot (falls back to current experiment)
# ---------------------------------------------------------------------------
def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


snapshot_dir = os.path.join(args.run_dir, "snapshot")
if os.path.exists(snapshot_dir):
    _phys = _load_module("snapshot_physics",
                         os.path.join(snapshot_dir, "physics.py"))
    _evol = _load_module("snapshot_evolution",
                         os.path.join(snapshot_dir, "evolution.py"))
    activate_outputs    = _phys.activate_outputs
    invest_liquidate    = _phys.invest_liquidate
    explore_physics     = _phys.explore_physics
    energy_physics      = _phys.energy_physics
    kill_random_chunk        = _evol.kill_random_chunk
    apply_radiation_mutation = _evol.apply_radiation_mutation
    get_energy_offset        = _evol.get_energy_offset
    # Use the snapshot's neat.config if available
    snap_config = os.path.join(snapshot_dir, "neat.config")
    if os.path.exists(snap_config):
        CONFIG_PATH = snap_config
    print(f"  [replay] Using snapshot from {snapshot_dir}")
else:
    from physics import (activate_outputs, invest_liquidate,
                         explore_physics, energy_physics)
    from evolution import (kill_random_chunk, apply_radiation_mutation,
                           get_energy_offset)
    print("  [replay] No snapshot found — using current experiment code")


# ---------------------------------------------------------------------------
# Discover checkpoints in the run directory
# ---------------------------------------------------------------------------
def discover_checkpoints(run_dir):
    dirs = glob.glob(os.path.join(run_dir, "checkpoint_*"))
    dirs = [d for d in dirs if re.search(r"checkpoint_(\d+)$", d)]
    return sorted(dirs, key=lambda d: int(re.search(r"checkpoint_(\d+)$", d).group(1)))


ckpt_dirs = discover_checkpoints(args.run_dir)
if not ckpt_dirs:
    print(f"ERROR: No checkpoints found in {args.run_dir}")
    sys.exit(1)

# Determine which checkpoint to start at
if args.step is not None:
    target = f"checkpoint_{args.step:07d}"
    matching = [d for d in ckpt_dirs if os.path.basename(d) == target]
    if not matching:
        available = [os.path.basename(d) for d in ckpt_dirs]
        print(f"ERROR: step {args.step} not found. Available: {available}")
        sys.exit(1)
    start_idx = ckpt_dirs.index(matching[0])
else:
    start_idx = len(ckpt_dirs) - 1  # latest

print(f"  [replay] {len(ckpt_dirs)} checkpoint(s) found in {args.run_dir}")
print(f"  [replay] Starting at: {os.path.basename(ckpt_dirs[start_idx])}")


# ---------------------------------------------------------------------------
# CoralReplayVis — CoralVis with a Replay navigation panel
# ---------------------------------------------------------------------------
class CoralReplayVis(Visualization):
    def __init__(self, substrate, evolver, vis_chs, run_dir, ckpt_dirs, ckpt_idx):
        super().__init__(substrate, vis_chs, panel_width=300)
        self.evolver    = evolver
        self.run_dir    = run_dir
        self.ckpt_dirs  = ckpt_dirs
        self.ckpt_idx   = ckpt_idx
        self.genome_stats = []
        self.fps_history  = []
        self.running      = False  # start paused; user enables to run forward
        self._load_pending = None  # index to load on next frame

    def load_ckpt(self, idx):
        """Queue a checkpoint load — executed between frames to avoid mid-render state."""
        self._load_pending = idx

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        px   = self.panel_x
        pw   = self.panel_wfrac
        n_chs = self.substrate.mem.shape[1]
        run_name = os.path.basename(self.run_dir)

        # -- Panel 1: Replay navigator — y=0.01..0.20 ------------------------
        with self.gui.sub_window("Replay", px, 0.01, pw, 0.20) as sw:
            # Truncate long run name to fit panel
            label = run_name if len(run_name) <= 28 else "..." + run_name[-25:]
            sw.text(label)
            sw.text(f"Ckpt {self.ckpt_idx+1}/{len(self.ckpt_dirs)}: "
                    f"step {self.evolver.timestep:,}")
            if sw.button("< Prev") and self.ckpt_idx > 0:
                self.load_ckpt(self.ckpt_idx - 1)
            if sw.button("> Next") and self.ckpt_idx < len(self.ckpt_dirs) - 1:
                self.load_ckpt(self.ckpt_idx + 1)
            self.running = not sw.checkbox("Paused", not self.running)

        # -- Panel 2: Display — y=0.22..0.57 ---------------------------------
        with self.gui.sub_window("Display", px, 0.22, pw, 0.36) as sw:
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
            self.paused = sw.checkbox("Pause render", self.paused)

        # -- Panel 3: Stats — y=0.59..0.99 ------------------------------------
        with self.gui.sub_window("Stats", px, 0.59, pw, 0.40) as sw:
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
            for i, n_cells, age in self.genome_stats[:6]:
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
    # Read grid shape from the first checkpoint's meta.json
    import json
    with open(os.path.join(ckpt_dirs[0], "meta.json")) as f:
        meta = json.load(f)
    shape = tuple(meta["shape"])

    substrate = Substrate(shape, torch.float32, DEVICE, CHANNELS)
    substrate.malloc()

    evolver = SpaceEvolver(CONFIG_PATH, substrate, KERNEL, DIR_ORDER,
                           SENSE_CHS, ACT_CHS)

    # Load the starting checkpoint (restores substrate + RNG state)
    evolver.load_checkpoint(ckpt_dirs[start_idx])

    vis = CoralReplayVis(substrate, evolver, ["energy", "infra", "rot"],
                         args.run_dir, ckpt_dirs, start_idx)

    inds = substrate.ti_indices[None]
    step_times = []
    fps_window = []
    fps_print_interval = 2.0
    last_fps_time = time.time()

    print(f"\nReplay | {shape[0]}x{shape[1]} | {args.backend}/{args.device}")
    print(f"  Run : {args.run_dir}")
    print(f"  Ckpt: {os.path.basename(ckpt_dirs[start_idx])}")
    print("-" * 60)
    print("  < Prev / > Next to navigate checkpoints")
    print("  Uncheck 'Paused' in the Replay panel to run forward")
    print("-" * 60)

    # Start paused — user explicitly unpauses to run forward
    vis.running = False
    step = evolver.timestep

    while vis.window.running:
        # Apply any pending checkpoint load before physics
        if vis._load_pending is not None:
            new_idx = vis._load_pending
            vis._load_pending = None
            evolver.load_checkpoint(ckpt_dirs[new_idx])
            vis.ckpt_idx = new_idx
            step = evolver.timestep
            print(f"  [loaded] {os.path.basename(ckpt_dirs[new_idx])}")

        if vis.running:
            t_step = time.perf_counter()

            sync()
            cw, cb = evolver.get_combined_weights()
            out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
            apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                     cw, cb, evolver.dir_kernel, evolver.dir_order,
                                     substrate.ti_indices)
            substrate.mem[0, evolver.act_chinds] = out_mem
            sync()

            activate_outputs(substrate)
            invest_liquidate(substrate)
            explore_physics(substrate, evolver.kernel, evolver.dir_order)
            energy_physics(substrate, evolver.kernel, max_infra=10, max_energy=1.5)

            alive = (substrate.mem[0, inds.infra] + substrate.mem[0, inds.energy]) > 0.05
            substrate.mem[0, inds.genome].masked_fill_(~alive, -1)

            evolver.energy_offset = get_energy_offset(step)
            evolver.ages = [a + 1 for a in evolver.ages]
            offset = evolver.energy_offset
            substrate.mem[0, inds.energy].add_(
                torch.randn_like(substrate.mem[0, inds.energy]).add_(offset).mul_(0.1))
            substrate.mem[0, inds.infra].add_(
                torch.randn_like(substrate.mem[0, inds.infra]).add_(offset).mul_(0.1))
            substrate.mem[0, inds.energy].clamp_(0.01, 100)
            substrate.mem[0, inds.infra].clamp_(0.01, 100)

            if step % 50 == 0:
                kill_random_chunk(evolver, 5)

            if step % 50 == 0 and step > 0:
                apply_radiation_mutation(evolver, 5)

            evolver.timestep = step + 1
            step += 1
            dt = time.perf_counter() - t_step
            step_times.append(dt)
            fps_window.append(dt)

            now = time.time()
            if now - last_fps_time >= fps_print_interval:
                fps = len(fps_window) / sum(fps_window)
                alive_n = (substrate.mem[0, inds.genome] >= 0).sum().item()
                vis.fps_history.append(fps)
                print(f"  step {step:5d} | {fps:5.1f} FPS | "
                      f"alive: {alive_n:,} | genomes: {len(evolver.genomes)}")
                fps_window = []
                last_fps_time = now

        vis.update()


if __name__ == "__main__":
    main()
