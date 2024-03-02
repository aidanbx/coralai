import os
import torch
import taichi as ti

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.evolution.organism_cppn import OrganismCPPN
from coralai.instances.nca.nca_organism_cnn import NCAOrganismCNN
from coralai.evolution.evolvable_organism import EvolvableOrganism
from coralai.evolution.run_things import run_cnn, run_evolvable

def define_substrate(shape, n_hidden_channels, torch_device):
    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=n_hidden_channels, dtype=ti.f32),
        },
    )
    substrate.malloc()
    return substrate

def main():
    shape = (100, 100)
    n_hidden_channels = 2
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = "coralai/instances/nca/nca_neat.config"
    config_path = os.path.join(local_dir, config_filename)

    substrate = define_substrate(shape, n_hidden_channels, torch_device)
    kernel = [[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]]
    
    sense_chs = ['rgb', 'hidden']
    act_chs = ['rgb', 'hidden']

    genome_key = 0
    genome_map = torch.zeros(shape[0], shape[1], dtype=torch.int32, device=torch_device)
    vis = Visualization(substrate, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])

    organism_cnn = NCAOrganismCNN(substrate, kernel, sense_chs, act_chs, torch_device)
    organism_rnn = EvolvableOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_cppn = OrganismCPPN(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism = organism_rnn

    organism.set_genome(genome_key, organism.gen_random_genome())
    organism.create_torch_net()

    # run_evolvable(vis, substrate, organism, genome_map)
    run_cnn(vis, organism_cnn, substrate)


if __name__ == "__main__":
    main()
