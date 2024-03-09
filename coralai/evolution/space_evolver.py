import copy
from datetime import datetime
import os
import random
import numpy as np
from neat.reporting import ReporterSet
from neat.reporting import BaseReporter
import pickle

import torch
import neat
from neat.six_util import iteritems, itervalues

import taichi as ti
import torch.nn as nn


from coralai.instances.coral.coral_physics import invest_liquidate, explore_physics, energy_physics, activate_outputs, apply_weights_and_biases

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import identity_activation
from pytorch_neat.linear_net import LinearNet
from ..substrate.nn_lib import ch_norm

@ti.data_oriented
class SpaceEvolver():
    def __init__(self, config_path, substrate, kernel, dir_order, sense_chs, act_chs):
        torch_device = substrate.torch_device
        self.torch_device = torch_device
        self.substrate = substrate
        self.reporters = ReporterSet()
        self.substrate = substrate
        self.kernel = torch.tensor(kernel, device=torch_device)
        self.dir_kernel = self.kernel[1:] # for directional operations
        # Represents the order in which the kernel should be applied
        # in this case, in alternating closest vectors from a direction
        self.dir_order = torch.tensor(dir_order, device=torch_device)
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)


        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        self.timestep = 0
        self.out_mem = None
        self.energy_offset = 0.0

        self.genomes = []
        self.ages = []
        self.combined_weights = []
        self.combined_biases = []
        self.init_population()
        self.init_substrate(self.genomes)
        self.time_last_cull = 0


        self.add_reporter(neat.StdOutReporter(True))
        self.add_reporter(neat.StatisticsReporter())

        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        self.checkpoint_dir = f'history/space_evolver_run_{current_datetime}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_prefix_full = os.path.join(self.checkpoint_dir, f"checkpoint")
        self.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix_full))
        # Copy config from config_path to checkpoint_dir
        with open(config_path, "r") as f:
            config_str = f.read()
        with open(os.path.join(self.checkpoint_dir, "neat_config"), "w") as f:
            f.write(config_str)
        self.substrate.save_metadata_to_json(filepath = os.path.join(self.checkpoint_dir, "sub_meta"))



    def run(self, n_timesteps, vis, n_rad_spots, radiate_interval, cull_max_pop, cull_interval=100):
        timestep = 0
        while timestep < n_timesteps and vis.window.running:
            combined_weights = torch.stack(self.combined_weights, dim=0)
            combined_biases = torch.stack(self.combined_biases, dim=0)
            self.step_sim(combined_weights, combined_biases)
            self.report_if_necessary(timestep)
            vis.update()
            if timestep % radiate_interval == 0:
                self.apply_radiation_mutation(n_rad_spots)
                print("RADIATING")
            if vis.next_generation:
                vis.next_generation = False
                break
            if len(self.genomes) > cull_max_pop and (self.timestep - self.time_last_cull) > cull_interval:
                # self.cull_genomes(cull_cell_thresh, cull_age_thresh)
                # if len(self.genomes) > cull_max_pop:
                self.reduce_population_to_threshold(cull_max_pop)
            timestep += 1
            self.timestep = timestep

    
    def step_sim(self, combined_weights, combined_biases):
        inds = self.substrate.ti_indices[None]
        self.forward(combined_weights, combined_biases)
        self.energy_offset = self.get_energy_offset(self.timestep)
        self.ages = [age + 1 for age in self.ages]
        # self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + self.energy_offset) * 0.1
        # self.substrate.mem[0, inds.infra] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + self.energy_offset) * 0.1
        self.substrate.mem[0, inds.energy] = torch.clamp(self.substrate.mem[0, inds.energy], 0.01, 100)
        self.substrate.mem[0, inds.infra] = torch.clamp(self.substrate.mem[0, inds.infra], 0.01, 100)
        if self.timestep % 10 == 0:
            self.kill_random_chunk(5)
            self.grow_random_chunk(5)


    def kill_random_chunk(self, radius):
        inds = self.substrate.ti_indices[None]
        x = np.random.randint(0, self.substrate.w)
        y = np.random.randint(0, self.substrate.h)
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                self.substrate.mem[0, inds.genome, i%self.substrate.w, j%self.substrate.h] = -1
                self.substrate.mem[0, inds.infra, i%self.substrate.w, j%self.substrate.h] = 0.01
                self.substrate.mem[0, inds.energy, i%self.substrate.w, j%self.substrate.h] = 0.01


    def grow_random_chunk(self, radius, energy_val = 5, infra_val = 5):
        inds = self.substrate.ti_indices[None]
        x = np.random.randint(0, self.substrate.w)
        y = np.random.randint(0, self.substrate.h)
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                self.substrate.mem[0, inds.infra, i%self.substrate.w, j%self.substrate.h] = energy_val
                self.substrate.mem[0, inds.energy, i%self.substrate.w, j%self.substrate.h] = infra_val


    def forward(self, weights, biases):
        inds = self.substrate.ti_indices[None]
        out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
        apply_weights_and_biases(
            self.substrate.mem, out_mem,
            self.sense_chinds,
            weights, biases,
            self.dir_kernel, self.dir_order,
            self.substrate.ti_indices)
        self.substrate.mem[0, self.act_chinds] = out_mem
        self.apply_physics()
    


    def apply_physics(self):
        inds = self.substrate.ti_indices[None]
        # self.substrate.mem[0, inds.energy, self.substrate.w//2, self.substrate.h//2] += 10
        activate_outputs(self.substrate)
        invest_liquidate(self.substrate)
        explore_physics(self.substrate, self.kernel, self.dir_order)
        energy_physics(self.substrate, self.kernel, max_infra=10, max_energy=1.5)

        self.substrate.mem[0, inds.genome] = torch.where(
            (self.substrate.mem[0, inds.infra] + self.substrate.mem[0, inds.energy]) > 0.05,
            self.substrate.mem[0, inds.genome],
            -1
        )


    def produce_alternating_order(self, len):
        order = []
        ind = 0
        i = 0
        while i < len:
            order.append(ind)
            if ind > 0:
                ind = -ind
            else:
                ind = (-ind + 1)
            i += 1
        return torch.tensor(order, device = self.torch_device)
    

    @ti.kernel
    def replace_genomes(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                        genome_transitions: ti.types.ndarray(), ti_indices: ti.template()):
        inds = ti_indices[None]
        for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
            if mem[0, inds.genome, i, j] < 0:
                out_mem[i, j] = mem[0, inds.genome, i, j]
            else:
                out_mem[i, j] = genome_transitions[int(mem[0, inds.genome, i, j])]

    def reduce_population_to_threshold(self, max_population):
        print(f"REDUCING pop to max of {max_population} from current size: {len(self.genomes)}")
        if len(self.genomes) <= max_population:
            print("Population within threshold. No reduction needed.")
            return

        inds = self.substrate.ti_indices[None]
        genome_cell_counts = [(i, self.substrate.mem[0, inds.genome].eq(i).sum().item()) for i in range(len(self.genomes))]
        # Sort genomes by cell count (ascending) to identify those with the lowest count
        sorted_genomes_by_cell_count = sorted(genome_cell_counts, key=lambda x: x[1], reverse=True)
        new_genomes = []
        new_ages = []
        new_combined_weights = []
        new_combined_biases = []
        genome_transitions = [None] * len(self.genomes)
        for i in range(len(sorted_genomes_by_cell_count)):
            index_of_genome = sorted_genomes_by_cell_count[i][0]
            if i > max_population:
                print(f"KILLING {index_of_genome}")
                genome_transitions[index_of_genome] = -1
            else:
                new_genomes.append(self.genomes[index_of_genome])
                new_ages.append(self.ages[index_of_genome])
                new_combined_weights.append(self.combined_weights[index_of_genome])
                new_combined_biases.append(self.combined_biases[index_of_genome])
                genome_transitions[index_of_genome] = len(new_genomes)
        genome_transitions = torch.tensor(genome_transitions, device = self.torch_device)
        out_mem = torch.zeros_like(self.substrate.mem[0, inds.genome])
        self.replace_genomes(self.substrate.mem, out_mem, genome_transitions, self.substrate.ti_indices)
        self.substrate.mem[0, inds.genome] = out_mem 

        self.genomes = new_genomes
        self.ages = new_ages
        self.combined_weights = new_combined_weights
        self.combined_biases = new_combined_biases
        self.time_last_cull = self.timestep
        print(f"\tPop size after reduction: {len(self.genomes)}")
        if len(self.genomes) == 0:
            print("NO GENOMES LEFT. REINITIALIZING")
            self.genomes = self.init_population()
            self.init_substrate(self.genomes)


    def load_checkpoint(self, folderpath):
        config_path = os.path.join(folderpath, "..", "neat_config")
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_path)
        self.load_genomes(os.path.join(folderpath, "genomes"))
        with open(os.path.join(folderpath, "ages"), 'rb') as f:
            self.ages = pickle.load(f)

        self.combined_biases = []
        self.combined_weights = []
        for genome in self.genomes:
            genome.fitness = 0.0
            net = self.create_torch_net(genome)
            self.combined_biases.append(net.biases)
            self.combined_weights.append(net.weights) 
        loaded_mem = torch.load(os.path.join(folderpath, "sub_mem"))
        self.substrate.mem = loaded_mem

        
    def save_genomes(self, genomes, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.genomes, f)
    
    def load_genomes(self, filename):
        with open(filename, 'rb') as f:
            genomes = pickle.load(f)
        return genomes
    
    def report_if_necessary(self, timestep):
        if timestep % 500 == 0:
            timestep_dir = os.path.join(self.checkpoint_dir, f"step_{timestep}")
            os.makedirs(timestep_dir, exist_ok=True)
            self.substrate.save_mem_to_pt(filepath = os.path.join(timestep_dir, "sub_mem"))
            self.save_genomes(self.genomes, os.path.join(timestep_dir, "genomes"))
            with open(os.path.join(timestep_dir, "ages"), 'wb') as f:
                pickle.dump(self.ages, f)

        # for i in range(len(self.genomes)):
        #     # org['genome'].fitness += self.substrate.mem[0, inds.genome].eq(i).sum().item()
        #     self.genomes[i].fitness = fitness_function(self.genomes[i], i)
        # self.reporters.start_generation(self.generation)

        # # Evaluate all genomes using the user-provided function.
        # fitness_function(list(iteritems(self.population)), self.config)

        # # Gather and report statistics.
        # best = None
        # for g in itervalues(self.population):
        #     if best is None or g.fitness > best.fitness:
        #         best = g

        # self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # # # Create the next generation from the current generation.
        # # self.population = self.reproduction.reproduce(self.config, self.species,
        # #                                               self.config.pop_size, self.generation)

        # # Check for complete extinction.
        # if not self.species.species:
        #     self.reporters.complete_extinction()
        #     # TODO: re-seed env

        # # Divide the new population into species.
        # self.species.speciate(self.config, self.population, self.generation)

        # self.reporters.end_generation(self.config, self.population, self.species)

        # self.generation += 1

        # if self.config.no_fitness_termination:
        #     self.reporters.found_solution(self.config, self.generation, self.best_genome)

        # return self.best_genome
    
    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)    

    def init_population(self):
        genomes = []
        for i in range(self.neat_config.pop_size):
            genome = neat.DefaultGenome(str(i))
            genome.configure_new(self.neat_config.genome_config)
            self.add_organism_get_key(genome)

        return genomes


    def init_substrate(self, genomes):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome] = torch.where(
            torch.rand_like(self.substrate.mem[0, inds.genome]) > 0.95,
            torch.randint_like(self.substrate.mem[0, inds.genome], 0, len(genomes)),
            -1
        )
        self.substrate.mem[0, inds.energy, ...] = 1.0
        self.substrate.mem[0, inds.infra, ...] = 1.0
        self.substrate.mem[0, inds.rot] = torch.randint_like(self.substrate.mem[0, inds.rot], 0, self.dir_kernel.shape[0])


    def add_organism_get_key(self, genome):
        # TODO: implement culling and memory consolidation
        self.genomes.append(genome)
        net = self.create_torch_net(genome)
        self.combined_weights.append(net.weights)
        self.combined_biases.append(net.biases)
        self.ages.append(0)
        return len(self.combined_biases) - 1
    

    def set_chunk(self, genome_key, x, y, radius):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome, x%self.substrate.w, y%self.substrate.h] = genome_key
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                self.substrate.mem[0, inds.genome, i%self.substrate.w, j%self.substrate.h] = genome_key        


    def get_energy_offset(self, timestep, repeat_steps=50, amplitude=1, positive_scale=1, negative_scale=1):
        frequency = (2 * np.pi) / repeat_steps
        value = amplitude * np.sin(frequency * timestep)
        if value > 0:
            return value * positive_scale
        else:
            return value * negative_scale

    
    def apply_radiation_mutation(self, n_spots, spot_live_radius=2, spot_dead_radius=4):
        inds = self.substrate.ti_indices[None]
        xs = torch.randint(0, self.substrate.w, (n_spots,))
        ys = torch.randint(0, self.substrate.h, (n_spots,))
        
        for i in range(n_spots):
            genome_key = int(self.substrate.mem[0, inds.genome, xs[i], ys[i]].item())
            rand_genome_key = torch.randint(0, len(self.genomes), (1,))
            self.set_chunk(-1, xs[i], ys[i], spot_dead_radius)
            if genome_key < 0:
                self.set_chunk(rand_genome_key, xs[i], ys[i], spot_live_radius)
            else:
                if random.random() < 0.5:
                    new_genome = copy.deepcopy(self.genomes[genome_key])
                    new_genome.mutate(self.neat_config.genome_config)
                    new_genome_key = self.add_organism_get_key(new_genome)
                    self.set_chunk(new_genome_key, xs[i], ys[i], spot_live_radius)
                else: 
                    new_genome = neat.DefaultGenome(str(len(self.genomes)))
                    self.genomes[genome_key].fitness = 0.0
                    self.genomes[rand_genome_key].fitness = 0.0
                    new_genome.configure_crossover(self.genomes[genome_key], self.genomes[rand_genome_key], self.neat_config)
                    new_genome_key = self.add_organism_get_key(new_genome)
                    self.set_chunk(new_genome_key, xs[i], ys[i], spot_live_radius)


    def create_torch_net(self, genome):
        input_coords = []
        # TODO: adjust for direcitonal kernel
        for ch in range(self.n_senses):
             input_coords.append([0, 0, self.sense_chinds[ch]])
             for offset_i in range(self.dir_order.shape[0]):
                offset_x = self.dir_kernel[self.dir_order[offset_i], 0]
                offset_y = self.dir_kernel[self.dir_order[offset_i], 1]
                input_coords.append([offset_x, offset_y, self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])

        net = LinearNet.create(
            genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.0,
            weight_max=3.0,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.torch_device,
        )
        return net
    

    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum

    # def cull_genomes(self, n_cells_thresh, age_thresh):
    #     print(f"CULLING pop of size: {len(self.genomes)}")
    #     inds = self.substrate.ti_indices[None]
    #     new_genomes = []
    #     new_ages = []
    #     new_combined_weights = []
    #     new_combined_biases = []
    #     for i in range(len(self.genomes)):
    #         where_i = self.substrate.mem[0, inds.genome].eq(i)
    #         n_cells = where_i.sum().item()
    #         print(f"\tGenome {i} is {self.ages[i]} steps old and has {n_cells} cells")
    #         if n_cells == 0 or (self.ages[i] > age_thresh and n_cells < n_cells_thresh):
    #             print(f"\t\tCulling genome {i}")
    #             self.substrate.mem[0, inds.genome] = torch.where(where_i, -1, self.substrate.mem[0, inds.genome])
    #         else:
    #             self.substrate.mem[0, inds.genome]= torch.where(where_i, len(new_genomes), self.substrate.mem[0, inds.genome])
    #             new_genomes.append(self.genomes[i])
    #             new_ages.append(self.ages[i])
    #             new_combined_weights.append(self.combined_weights[i])
    #             new_combined_biases.append(self.combined_biases[i])
    #     self.genomes = new_genomes
    #     self.ages = new_ages
    #     self.combined_weights = new_combined_weights
    #     self.combined_biases = new_combined_biases
    #     print(f"\tPop size after culling: {len(self.genomes)}")
    #     if len(self.genomes) == 0:
    #         print("Population extinct")
    #         self.init_population()
    #         self.init_substrate(self.genomes)
    #         self.timestep = 0