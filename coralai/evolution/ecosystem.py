import torch
import neat

class Ecosystem():
    def __init__(self, substrate, create_organism, apply_physics, initial_size):
        self.substrate = substrate
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome] = torch.full_like(self.substrate.mem[0, inds.genome], 0)
        self.population = {}
        self.next_free_genome_key = 0
        for _ in range(initial_size):
            org = create_organism(genome_key=self.next_free_genome_key)
            self.population[self.next_free_genome_key] = {"org": org, "infra": 0.1}
            self.next_free_genome_key += 1
        

        self.create_organism = create_organism
        self.apply_physics = apply_physics
        self.min_size = initial_size
        self.time_step = 0

    
    def sexual_reproduction(self, merging_cell_coords, incoming_genome_matrix):
        inds = self.substrate.ti_indices[None]
        for i, (x, y) in enumerate(merging_cell_coords):
            parent_old = self.population[self.mem.substrate[0, inds.genome,x,y]]['org'].genome
            parent_incoming = self.population[incoming_genome_matrix[x, y]]['org'].genome
            child_genome = neat.DefaultGenome(str(self.next_free_genome_key))
            child_genome.configure_crossover(parent_old, parent_incoming, parent_old.neat_config)
            child_organism = self.create_organism(genome_key=self.next_free_genome_key, genome=child_genome)
            self.population[self.next_free_genome_key] = {"org": child_organism, "infra": 0.1}
            self.substrate.mem[0, inds.genome, x, y] = self.next_free_genome_key
            self.next_free_genome_key += 1


    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum


    def update_population_infra_sum(self):
        for genome_key in self.population.keys():
            infra_sum = self.get_genome_infra_sum(genome_key)
            self.population[genome_key]["infra"] = infra_sum


    def get_random_genome_keys(self, n_genomes):
        inds = self.substrate.ti_indices[None]
        infras = torch.tensor([org_info['infra'] for org_info in self.population.values()], dtype=torch.float32)
        infra_sum = infras.sum()
        if infra_sum != 0:
            infra_probs = infras / infra_sum
        else:
            # Create a uniform distribution if infra_sum is 0
            infra_probs = torch.ones_like(infras) / len(infras)
        selected_index = torch.multinomial(infra_probs, n_genomes, replacement=True)
        selected_genomes = [list(self.population.keys())[i] for i in selected_index]
        return selected_genomes


    def sew_seeds(self, n_seeds):
        inds = self.substrate.ti_indices[None]
        selected_genome_keys = self.get_random_genome_keys(n_seeds)
        random_x_coords = torch.randint(0, self.substrate.w, (n_seeds,))
        random_y_coords = torch.randint(0, self.substrate.h, (n_seeds,))
        for i in range(n_seeds):
            self.substrate.mem[0, inds.genome, random_x_coords[i], random_y_coords[i]] = selected_genome_keys[i]
            self.substrate.mem[0, inds.infra, random_x_coords[i], random_y_coords[i]] += 2


    def apply_radiation(self, n_radiation_spots):
        inds = self.substrate.ti_indices[None]
        selected_genome_keys = self.get_random_genome_keys(n_radiation_spots)
        random_x_coords = torch.randint(0, self.substrate.w, (n_radiation_spots,))
        random_y_coords = torch.randint(0, self.substrate.h, (n_radiation_spots,))
        for i in range(n_radiation_spots):
            new_genome = self.population[selected_genome_keys[i]]['org'].mutate()
            new_organism = self.create_organism(genome_key=self.next_free_genome_key, genome=new_genome)
            self.population[self.next_free_genome_key] = {"org": new_organism, "infra": 0.1}
            self.substrate.mem[0, inds.genome, random_x_coords[i], random_y_coords[i]] = self.next_free_genome_key
            self.substrate.mem[0, inds.infra, random_x_coords[i], random_y_coords[i]] += 2
            self.next_free_genome_key += 1
            

    def update(self, seed_interval, seed_volume, radiation_interval, radiation_volume):
        # if self.time_step % seed_interval == 0:
        #     self.sew_seeds(seed_volume)
        # if self.time_step % radiation_interval == 0:
        #     self.apply_radiation(radiation_volume)
        for org_info in self.population.values():
            org_info['org'].forward()
        # self.apply_physics()

