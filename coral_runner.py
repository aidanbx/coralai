import os
import torch
import taichi as ti
from coralai.substrate.substrate import Substrate
from coralai.instances.coral.coral_physics import apply_actuators
from coralai.instances.coral.coral_vis import CoralVis
from coralai.instances.coral.coral_organism import CoralOrganism
from coralai.evolution.run_things import run_cnn, run_evolvable


def define_substrate(shape, n_hidden_channels, torch_device):
    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "energy": ti.f32,
            "infra": ti.f32,
            "last_move": ti.f32,
            "invest_act": ti.f32,
            "liquidate_act": ti.f32,
            "explore_act": ti.f32,
            "com": ti.types.vector(n=n_hidden_channels, dtype=ti.f32),
        },
    )
    substrate.malloc()
    return substrate


def main():
    shape = (400, 400)
    n_hidden_channels = 8
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = "coralai/instances/coral/coral_neat.config"
    config_path = os.path.join(local_dir, config_filename)

    substrate = define_substrate(shape, n_hidden_channels, torch_device)
    kernel = [[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]]
    
    sense_chs = ['energy', 'infra', 'last_move', 'com']
    act_chs = ['invest_act', 'liquidate_act', 'explore_act', 'com']

    genome_key = 0
    genome_map = torch.zeros(shape[0], shape[1], dtype=torch.int32, device=torch_device)
    
    organism = CoralOrganism(substrate, kernel, sense_chs, act_chs, torch_device)
    vis = CoralVis(substrate, ['energy', 'infra', 'last_move'])

    while vis.window.running:
        apply_actuators(substrate, organism.forward(substrate.mem))

        vis.update()
        if vis.mutating:
            organism.mutate(vis.perturbation_strength)

if __name__ == "__main__":
    main()
