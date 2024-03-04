import torch

class Ecosystem():
    def __init__(self, susbtrate, create_organism, apply_physics, initial_size, seed_rate):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome] = torch.full_like(self.substrate.mem[0, inds.genome], -1)
        self.population = {}

        for i in range(initial_size):
            org = create_organism(genome_key=i)
            self.population[i] = {"org": org, "infra": 0}
        
        self.apply_physics = apply_physics
        self.min_size = initial_size
        self.seed_rate = seed_rate
        self.time_step = 0


    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum


    def update_population_infra_sum(self):
        for genome_key in self.population.keys():
            infra_sum = self.get_genome_infra_sum(genome_key)
            self.population[genome_key]["infra"] = infra_sum


    def update(self):
        for organism in self.population.values():
            organism.forward()


    def reproduce(self, merging_cells, new_genomes):
        if len(merging_cells) > 0:
            inds = self.substrate.ti_indices[None]

