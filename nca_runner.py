import os
import torch
import taichi as ti

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.instances.nca.nca_organism_cnn import NCAOrganismCNN
from coralai.instances.nca.nca_organism_hyper import NCAOrganismHyper
from coralai.evolution.evolvable_organism import EvolvableOrganism
from coralai.evolution.organism_cppn import OrganismCPPN

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
    genome_map = torch.zeros(shape, dtype=torch.int32, device=torch_device)
    num_cells_to_activate = 10
    for _ in range(num_cells_to_activate):
        x = torch.randint(0, shape[0], (1,))
        y = torch.randint(0, shape[1], (1,))
        genome_map[x, y] = 1
    # genome_map = torch.randint(0, 10, shape, dtype=torch.int32, device=torch_device)
    vis = Visualization(substrate, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])

    organism_cnn = NCAOrganismCNN(substrate, kernel, sense_chs, act_chs, torch_device)
    organism_rnn = EvolvableOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_cppn = OrganismCPPN(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_hyper = NCAOrganismHyper(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism = organism_hyper

    if organism.is_evolvable:
        organism.set_genome(genome_key, organism.gen_random_genome())
        organism.create_torch_net()
    
    while vis.window.running:
        for e in vis.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                vis.substrate.mem *= 0.0
                organism.set_genome(genome_key, organism.gen_random_genome())
                organism.create_torch_net()
        if organism.is_evolvable:
            organism.forward(substrate.mem, genome_map)
        else:
            organism.forward(substrate.mem)
        vis.update()
        if vis.mutating:
            new_genome = organism.mutate(vis.perturbation_strength)
            if organism.is_evolvable:
                organism.set_genome(organism.genome_key, new_genome) # mutates all cells at once
                organism.create_torch_net()
                print(organism.net.weights)


if __name__ == "__main__":
    main()
