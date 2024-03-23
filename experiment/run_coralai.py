import os
import torch
import neat
import taichi as ti
from coralai.substrate import Substrate
from coralai.evolver_old import SpaceEvolver
from coralai.visualization import Visualization


def main(config_filename, channels, shape, kernel, dir_order, sense_chs, act_chs, torch_device):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()

    inds = substrate.ti_indices[None]

    space_evolver = SpaceEvolver(config_path, substrate, kernel, dir_order, sense_chs, act_chs)
    # checkpoint_dir = os.path.join(local_dir, "history", "space_evolver_run_240310-0027_01", "step_900")
    # space_evolver.load_checkpoint(folderpath=checkpoint_dir)
    vis = CoralVis(substrate, space_evolver, ["energy", "infra", "genome_inv"])
    space_evolver.run(100000000, vis, n_rad_spots = 4, radiate_interval = 20,
                      cull_max_pop=1000, cull_interval=500)
    
    # checkpoint_file = os.path.join('history', 'NEAT_240308-0052_32', 'checkpoint4')
    # p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    # p.run(eval_vis, 10)


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename = "./coral_neat.config",
        channels = {
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=4, dtype=ti.f32) # no, forward, left, right
            ),
            "com": ti.types.struct(
                a=ti.f32,
                b=ti.f32,
                c=ti.f32,
                d=ti.f32
            ),
            "rot": ti.f32,
            "genome": ti.f32,
            "genome_inv": ti.f32
        },
        shape = (100, 100),
        kernel = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], # ccw
        dir_order = [0, -1, 1, -2], # forward (with rot), left of rot, right of rot, behind
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )
