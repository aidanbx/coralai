import os
import torch

import taichi as ti

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.evolution.organism_cppn import OrganismCPPN
from coralai.instances.minimal.minimal_organism_cnn import MinimalOrganismCNN
from coralai.evolution.evolvable_organism import EvolvableOrganism
from coralai.evolution.run_things import run_cnn, run_evolvable


def define_substrate(shape, torch_device):
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


def main():
    shape = (256, 256)
    torch_device = torch.device("mps")
    ti.init(ti.metal)

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = "coralai/instances/minimal/minimal_neat.config"
    config_path = os.path.join(local_dir, config_filename)

    substrate = define_substrate(shape, torch_device)
    kernel = [[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]]
    
    sense_chs = ['bw']
    act_chs = ['bw']
    
    genome_key = 0
    genome_map = torch.zeros(shape[0], shape[1], dtype=torch.int32, device=torch_device)
    vis = Visualization(substrate, ["bw"])

    organism_cnn = MinimalOrganismCNN(substrate, kernel, sense_chs, act_chs, torch_device)
    organism_rnn = EvolvableOrganism(config_path,substrate, kernel, sense_chs, act_chs, torch_device)
    organism_cppn = OrganismCPPN(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism = organism_rnn
    
    organism.set_genome(genome_key, organism.gen_random_genome())
    organism.create_torch_net()
    
    # run_evolvable(vis, substrate, organism, genome_map)
    run_cnn(vis, organism_cnn, substrate)


if __name__ == "__main__":
    main()
