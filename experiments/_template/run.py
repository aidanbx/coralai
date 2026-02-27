"""
Template experiment runner.

Copy this directory to experiments/<your_experiment>/, then:
1. Define your channel layout in CHANNELS below
2. Implement activate_outputs() and step_physics() in physics.py
3. Tune neat.config for your experiment
4. Run: python experiments/<your_experiment>/run.py --no-gui --steps 100

See experiments/coral/ for a complete working example.
"""

import os
import time
import argparse

import torch
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=0,
                    help="Steps to run then exit (0 = run until window closed)")
parser.add_argument("--shape", type=int, default=100)
parser.add_argument("--backend", type=str, default="metal",
                    choices=["cpu", "metal", "cuda", "vulkan"])
parser.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "mps", "cuda"])
parser.add_argument("--no-gui", action="store_true")
args = parser.parse_args()

ti.init({"cpu": ti.cpu, "metal": ti.metal,
         "cuda": ti.cuda, "vulkan": ti.vulkan}[args.backend])
DEVICE = torch.device(args.device)

from coralai.substrate import Substrate
from coralai.evolver import SpaceEvolver, apply_weights_and_biases
from coralai.visualization import Visualization
from physics import activate_outputs, step_physics

# ---------------------------------------------------------------------------
# 1. Define your channel layout
# ---------------------------------------------------------------------------
CHANNELS = {
    "energy":  ti.f32,
    "infra":   ti.f32,
    "acts":    ti.types.struct(
        invest=ti.f32,
        liquidate=ti.f32,
        explore=ti.types.vector(n=4, dtype=ti.f32),
    ),
    "com":     ti.types.struct(a=ti.f32, b=ti.f32, c=ti.f32, d=ti.f32),
    "rot":     ti.f32,
    "genome":  ti.f32,
}

# 2. Define the spatial kernel (neighborhood shape)
KERNEL    = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]  # von Neumann
DIR_ORDER = [0, -1, 1]   # forward, left, right

# 3. Which channels the network reads and writes
SENSE_CHS = ["energy", "infra", "com"]
ACT_CHS   = ["acts", "com"]

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neat.config")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    shape = (args.shape, args.shape)
    substrate = Substrate(shape, torch.float32, DEVICE, CHANNELS)
    substrate.malloc()

    evolver = SpaceEvolver(CONFIG_PATH, substrate, KERNEL, DIR_ORDER,
                           SENSE_CHS, ACT_CHS)

    vis = None if args.no_gui else Visualization(substrate, ["energy", "infra"])
    inds = substrate.ti_indices[None]
    max_steps = args.steps if args.steps > 0 else 10 ** 9

    for step in range(max_steps):
        cw, cb = evolver.get_combined_weights()
        out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
        apply_weights_and_biases(substrate.mem, out_mem, evolver.sense_chinds,
                                 cw, cb, evolver.dir_kernel, evolver.dir_order,
                                 substrate.ti_indices)
        substrate.mem[0, evolver.act_chinds] = out_mem

        activate_outputs(substrate)
        step_physics(substrate)

        # Death
        alive = (substrate.mem[0, inds.infra] + substrate.mem[0, inds.energy]) > 0.05
        substrate.mem[0, inds.genome].masked_fill_(~alive, -1)

        evolver.ages = [a + 1 for a in evolver.ages]
        evolver.timestep = step + 1

        if step % 50 == 0 and step > 0:
            evolver.apply_radiation_mutation(5)

        if vis:
            vis.update()
            if not vis.window.running:
                break

if __name__ == "__main__":
    main()
