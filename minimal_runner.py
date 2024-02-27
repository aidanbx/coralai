import os
import torch
import neat

import taichi as ti
import configparser
from datetime import datetime


from coralai.substrate.substrate import Substrate
from coralai.instances.minimal.minimal_vis import MinimalVis
# from coralai.instances.minimal.minimal_organism_torch import MinimalOrganism
from coralai.instances.minimal.minimal_organism_rnn import MinimalOrganism

SHAPE = (256, 256)
torch_device = torch.device("mps")
ti.init(ti.metal)

def define_substrate(shape):
    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "bw": ti.f32,
        },
    )
    substrate.malloc()
    return substrate


def define_organism(substrate):
    config_filename = "./coralai/instances/minimal/minimal_neat.config"
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_filename)

    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)
    
    return MinimalOrganism(neat_config,
                           substrate,
                           sensors = ['bw'],
                           n_actuators = 1,
                           torch_device = substrate.torch_device)


def main():
    substrate = define_substrate(SHAPE)
    organism = define_organism(substrate)
    vis = MinimalVis(substrate, ["bw"])

    genome_key = 0
    genome_map = torch.zeros(SHAPE[0], SHAPE[1], dtype=torch.int32, device=torch_device)
    organism.set_genome(genome_key=genome_key)
    while vis.window.running:
        substrate.mem = organism.forward(substrate.mem, genome_map)
        vis.update()
        if vis.mutating:
            new_genome = organism.mutate(vis.perturbation_strength)
            organism.set_genome(genome_key, new_genome)
            # print(new_genome)


if __name__ == "__main__":
    main()
