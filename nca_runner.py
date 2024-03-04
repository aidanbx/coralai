import os
import torch
import taichi as ti

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.instances.nca.nca_organism_cnn import NCAOrganismCNN
from coralai.evolution.hyper_organism import HyperOrganism
from coralai.evolution.neat_organism import NeatOrganism
from coralai.evolution.cppn_organism import CPPNOrganism


def main(config_filename, channels, shape, kernel, sense_chs, act_chs, torch_device):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()

    genome_key = 0
    genome_map = torch.zeros(shape, dtype=torch.int32, device=torch_device)
    num_cells_to_activate = 10
    for _ in range(num_cells_to_activate):
        x = torch.randint(0, shape[0], (1,))
        y = torch.randint(0, shape[1], (1,))
        genome_map[x, y] = 1
    # genome_map = torch.randint(0, 10, shape, dtype=torch.int32, device=torch_device)
    vis = Visualization(substrate, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])

    organism_cnn = NCAOrganismCNN(substrate, kernel, sense_chs, act_chs, torch_device)
    organism_rnn = NeatOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_cppn = CPPNOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_hyper = HyperOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism = organism_hyper

    if organism.is_evolvable:
        organism.set_genome(genome_key, organism.gen_random_genome())
        organism.create_torch_net()
    
    while vis.window.running:
        if organism.is_evolvable:
            organism.forward(substrate.mem, genome_map)
        else:
            substrate.mem = organism.forward(substrate.mem)
        vis.update()
        if vis.mutating:
            new_genome = organism.mutate(vis.perturbation_strength)
            if organism.is_evolvable:
                organism.set_genome(organism.genome_key, new_genome) # mutates all cells at once
                organism.create_torch_net()
                print(organism.net.weights)
                vis.mutating=False


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename="coralai/instances/nca/nca_neat.config",
        channels={
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=2, dtype=ti.f32),
        },
        shape=(800, 800),
        kernel=[[-1,-1],[0,-1],
                [-1, 0],[0, 0],[1, 0],
                [0, 1],[1, 1]],
        sense_chs=['rgb', 'hidden'],
        act_chs=['rgb', 'hidden'],
        torch_device=torch_device
    )
