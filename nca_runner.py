import os
import torch
import taichi as ti
import torch.nn as nn

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.instances.nca.nca_organism_cnn import NCAOrganismCNN
from coralai.evolution.hyper_organism import HyperOrganism
from coralai.evolution.neat_organism import NeatOrganism
from coralai.evolution.cppn_organism import CPPNOrganism


def nca_activation(mem):
    mem = nn.ReLU()(mem)
    # Calculate the mean across batch and channel dimensions
    mean = mem.mean(dim=(0, 2, 3), keepdim=True)
    # Calculate the variance across batch and channel dimensions
    var = mem.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    # Normalize the input tensor
    mem.sub_(mean).div_(torch.sqrt(var + 1e-5))
    mem = torch.sigmoid(mem)
    return mem


def main(config_filename, channels, shape, kernel, sense_chs, act_chs, torch_device):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()

    genome_key = 0
    genome_map = torch.zeros(shape, dtype=torch.int32, device=torch_device)

    # genome_map = torch.randint(0, 10, shape, dtype=torch.int32, device=torch_device)
    vis = Visualization(substrate, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])

    organism_cnn = NCAOrganismCNN(substrate, kernel, sense_chs, act_chs, torch_device)
    organism_rnn = NeatOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_cppn = CPPNOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism_hyper = HyperOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
    organism = organism_hyper

    if organism.is_evolvable:
        organism.set_genome(genome_key, organism.gen_random_genome(genome_key))
        organism.create_torch_net()
    
    out_mem = torch.zeros_like(substrate.mem[0, organism.act_chinds])
    while vis.window.running:
        # substrate.mem += torch.randn_like(substrate.mem) * 0.1
        out_mem = organism.forward(out_mem, genome_map)
        substrate.mem[0, organism.act_chinds] = out_mem
        substrate.mem[0, organism.act_chinds] = nca_activation(substrate.mem[:, organism.act_chinds])
        vis.update()
        if vis.mutating:
            if organism.is_evolvable:
                new_genome = organism.mutate()
                organism.set_genome(organism.genome_key, new_genome) # mutates all cells at once
                organism.create_torch_net()
                print(organism.net.weights)
                print(organism.genome)
                vis.mutating=False


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename="coralai/instances/nca/nca_neat.config",
        channels={
            "genome": ti.f32,
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=2, dtype=ti.f32),
        },
        shape=(80, 80),
        kernel=[        [0,-1],
                [-1, 0],[0, 0],[1, 0],
                        [0, 1]],
        sense_chs=['rgb', 'hidden'],
        act_chs=['rgb', 'hidden'],
        torch_device=torch_device
    )
