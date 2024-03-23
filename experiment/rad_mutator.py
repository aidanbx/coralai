import copy
import torch
import neat
from random import random
from ..population import Population
from ..coralai_cor import CoralaiCor

def apply_radiation_mutation(cor: CoralaiCor, pop: Population, params):
    if params is None:
        params = {}
    n_spots = params.get("n_spots", 100)
    spot_live_radius = params.get("spot_live_radius", 2)
    spot_dead_radius = params.get("spot_dead_radius", 3)

    inds = cor.substrate.ti_indices[None]
    xs = torch.randint(0, cor.substrate.w, (n_spots,))
    ys = torch.randint(0, cor.substrate.h, (n_spots,))
    
    out_mem = cor.substrate.mem.clone()
    for i in range(n_spots):
        genome_key = int(cor.substrate.mem[0, inds.genome, xs[i], ys[i]].item())
        inds_genome = torch.tensor([inds.genome], device=cor.torch_device)
        vals = torch.tensor([0.0], device=cor.torch_device)
        x = xs[i].item()
        y = ys[i].item()
        vals[0] = -1

        rand_genome_key = torch.randint(0, len(pop.genomes), (1,))
        if genome_key < 0:
            cor.substrate.mem[0, inds.genome, x, y] = rand_genome_key
        else:
            if random.random() < 0.5:
                new_genome = copy.deepcopy(pop.genomes[genome_key])
                new_genome.mutate(cor.neat_config.genome_config)
                new_genome_key = pop.add_organism_get_key(new_genome)
                
                vals[0] = new_genome_key
                cor.set_chunk(out_mem, x, y, spot_live_radius, inds_genome, vals)
            else: 
                new_genome = neat.DefaultGenome(str(len(pop.genomes)))
                pop.genomes[genome_key].fitness = 0.0
                pop.genomes[rand_genome_key].fitness = 0.0
                new_genome.configure_crossover(pop.genomes[genome_key], pop.genomes[rand_genome_key], cor.neat_config)
                new_genome_key = pop.add_organism_get_key(new_genome)
                cor.substrate.mem[0, inds.genome, x, y] = new_genome_key