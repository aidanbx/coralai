from datetime import datetime
import os
import torch
import neat
import taichi as ti
import torch.nn as nn

from ..substrate.nn_lib import ch_norm

from coralai.instances.coral.coral_physics import invest_liquidate, explore_physics, energy_physics

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import relu_activation, sigmoid_activation, tanh_activation, identity_activation
from pytorch_neat.linear_net import LinearNet
from .neat_organism import NeatOrganism
from ..substrate.nn_lib import ch_norm

@ti.data_oriented
class NEATEvolver():
    def __init__(self, config_path, substrate, kernel, ind_of_middle, sense_chs, act_chs):
        self.substrate = substrate
        torch_device = substrate.torch_device
        self.substrate = substrate
        self.kernel = kernel
        self.torch_device = torch_device
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)

        self.ind_of_middle = ind_of_middle

        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        self.population = neat.Population(self.neat_config)
        
        self.time_step = 0
        self.out_mem = None
        self.energy_offset = 0.0

        # Create the population, which is the top-level object for a NEAT run.
        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())


        # Save the modified configuration in 'configs' folder with a specific name format
        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        checkpoint_dir = f'history/NEAT_{current_datetime}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix_full = os.path.join(checkpoint_dir, f"checkpoint")
        self.population.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix_full))

    
    def eval_genomes(self, genomes, config, n_timesteps, vis=None):
        organisms = []
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            organisms.append({"net": self.create_torch_net(genome, config), "genome": genome})

        self.run_sim(organisms, n_timesteps, vis)
        inds = self.substrate.ti_indices[None]
        for i in range(len(organisms)):
            org = organisms[i]
            org['genome'].fitness = self.substrate.mem[0, inds.genome].eq(i).sum().item()
            # org['genome'].fitness = (self.get_genome_infra_sum(i)).item()

    def run_sim(self, organisms, n_timesteps, vis = None):
        inds = self.substrate.ti_indices[None]
        # set 20% of cells to genome -1
        self.substrate.mem[0, inds.genome] = torch.where(
            torch.rand_like(self.substrate.mem[0, inds.genome]) > 0.99,
            torch.randint_like(self.substrate.mem[0, inds.genome], 0, len(organisms)),
            -1
        )

        combined_weights = torch.zeros(
            (len(organisms), 1, self.n_acts, self.n_senses * self.kernel.shape[0]), device=self.torch_device)
        combined_biases = torch.zeros(
            (len(organisms), 1, self.n_acts, 1), device=self.torch_device)
        
        for i in range(len(organisms)):
            combined_weights[i, 0] = organisms[i]["net"].weights
            combined_biases[i, 0] = organisms[i]["net"].biases

        out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
        for timestep in range(n_timesteps):
            self.step_sim(combined_weights, combined_biases, out_mem, timestep)
            if vis is not None:
                vis.update()

    
    def step_sim(self, combined_weights, combined_biases, out_mem, timestep):
        inds = self.substrate.ti_indices[None]
        self.apply_weights_and_biases(
            self.substrate.mem, out_mem,
            self.kernel, self.sense_chinds,
            combined_weights, combined_biases,
            inds.genome)
        self.substrate.mem[0, self.act_chinds] = out_mem
        self.substrate.mem[0, inds.com] = torch.sigmoid(nn.ReLU()(ch_norm(self.substrate.mem[:, inds.com])))
        self.energy_offset = self.get_energy_offset(timestep)
        self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + self.energy_offset) * 0.1
        self.substrate.mem[0, inds.infra] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + self.energy_offset) * 0.1
        self.apply_physics()
    

    def apply_physics(self):
        inds = self.substrate.ti_indices[None]
        invest_liquidate(self.substrate)
        explore_physics(self.substrate, self.kernel, self.ind_of_middle)
        energy_physics(self.substrate, self.kernel)


    def get_energy_offset(self, timestep, cycle_length=100, percent_day=0.5, night_intensity=1, day_intensity=1):
        cycle_completion = (timestep % cycle_length) / cycle_length # 0-1
        if cycle_completion < percent_day:
            return day_intensity
        else:
            return -night_intensity


    @ti.kernel
    def apply_weights_and_biases(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                                      kernel: ti.types.ndarray(), sense_chinds: ti.types.ndarray(),
                                      combined_weights: ti.types.ndarray(), combined_biases: ti.types.ndarray(),
                                      genome_ind: ti.i32):
        for i, j, act_k in ti.ndrange(mem.shape[2], mem.shape[3], out_mem.shape[0]):
            val = 0.0
            genome_key = int(mem[0, genome_ind, i, j])
            for sensor_n, neigh_m in ti.ndrange(sense_chinds.shape[0], kernel.shape[0]):
                neigh_x = (i + kernel[neigh_m, 0]) % mem.shape[2]
                neigh_y = (j + kernel[neigh_m, 1]) % mem.shape[3]
                val += (mem[0, sense_chinds[sensor_n], neigh_x, neigh_y] *
                        combined_weights[genome_key, 0, act_k, sensor_n])
            out_mem[act_k, i, j] = val + combined_biases[genome_key, 0, act_k, 0]


    def create_torch_net(self, genome, config):
        input_coords = []
        for offset in self.kernel:
            for ch in range(self.n_senses):
                input_coords.append([offset[0], offset[1], self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])

        net = LinearNet.create(
            genome,
            config,
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

    # def forward(self, out_mem, genome_map=None):
    #     with torch.no_grad():
    #         inds = self.substrate.ti_indices[None]

    #         if genome_map is None: 
    #             if 'genome' in self.substrate.channels:
    #                 genome_map = self.substrate.mem[0, inds.genome]
    #             else:
    #                 raise ValueError("hyper_organism: No genome map provided and no genome channel in substrate")
            
    #         cell_coords = self.get_cell_coords(genome_map)
    #         if cell_coords.shape[0] == 0:
    #             out_mem
            
    #         self.apply_weights_and_biases(self.substrate.mem, out_mem, cell_coords,
    #                   self.kernel, self.sense_chinds,
    #                   self.net.weights, self.net.biases)
            
    #         return out_mem

    # @ti.kernel
    # def apply_many_weights_and_biases(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
    #                                   kernel: ti.types.ndarray(), sense_chinds: ti.types.ndarray(),
    #                                   all_weights: ti.types.ndarray(), all_biases: ti.types.ndarray(),
    #                                   ti_inds: ti.template()):
    #     inds = ti_inds[None]
    #     for i, j, act_k in ti.ndrange(mem.shape[2], mem.shape[3], out_mem.shape[0]):
    #         val = 0.0
    #         ind_of_genome = 0
    #         for g_i in ti.ndrange(genome_keys.shape[0]):
    #             if genome_keys[g_i] == mem[0, inds.genome, i, j]:
    #                 ind_of_genome = g_i

    #         for sensor_n, off_m in ti.ndrange(sense_chinds.shape[0], kernel.shape[0]):
    #             neigh_x = (i + kernel[off_m, 0]) % mem.shape[2]
    #             neigh_y = (j + kernel[off_m, 1]) % mem.shape[3]
    #             val += mem[0, sense_chinds[sensor_n], neigh_x, neigh_y] * weights[0, act_j, sensor_n]
    #         out_mem[act_j, center_x, center_y] = val + biases[0, act_j, 0]


    # def sew_seeds(self, n_seeds):
    #     inds = self.substrate.ti_indices[None]
    #     selected_genome_keys = torch.randint(0, self.pop_size, (n_seeds,))
    #     random_x_coords = torch.randint(0, self.substrate.w, (n_seeds,))
    #     random_y_coords = torch.randint(0, self.substrate.h, (n_seeds,))
    #     for i in range(n_seeds):
    #         self.substrate.mem[0, inds.genome, random_x_coords[i], random_y_coords[i]] = selected_genome_keys[i]


    # def mutate(self, genome_key, report=False):
    #     new_genome = self.population[genome_key]['org'].mutate()
    #     new_organism = self.create_organism(genome_key=self.next_free_genome_key, genome=new_genome)
    #     self.population[self.next_free_genome_key] = {"org": new_organism, "infra": 0.1, "age": 0}
    #     self.next_free_genome_key += 1
    #     if report:
    #         print(f"Mutated genome {genome_key} to {self.next_free_genome_key-1}")
    #         print(f"New Genome: {new_genome}")
    #         print(f"Weights: {new_organism.net.weights}")
    #         print("--------")
    #     return self.next_free_genome_key-1


    # def apply_radiation(self, n_radiation_spots):
    #     inds = self.substrate.ti_indices[None]
    #     random_x_coords = torch.randint(0, self.substrate.w, (n_radiation_spots,))
    #     random_y_coords = torch.randint(0, self.substrate.h, (n_radiation_spots,))
    #     for i in range(n_radiation_spots):
    #         x = random_x_coords[i]
    #         y = random_y_coords[i]
    #         genome_at_loc = self.substrate.mem[0, inds.genome, x, y]
    #         if genome_at_loc != -1:
                
    #         new_genome_key = self.mutate(selected_genome_keys[i])
    #         coords = self.get_random_coords_of_genome(selected_genome_keys[i])
    #         if coords:  # Check if coords is not empty
    #             self.substrate.mem[0, inds.genome, coords[0][0], coords[0][1]] = new_genome_key
    
    
    # # def save_population(self):



    # def update(self, seed_interval=100, seed_volume=10, radiation_interval=500, radiation_volume=10):
    #     # self.update_population_infra_sum()

    #     if self.time_step % seed_interval == 0:
    #         self.sew_seeds(seed_volume)

    #     if self.time_step % radiation_interval == 0:
    #         self.apply_radiation(radiation_volume)

    #     if self.out_mem is None:
    #         self.out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
    #     else:
    #         self.out_mem[:] = 0.0

    #     genomes_to_remove = []
    #     for genome_key, org_info in self.population.items():
    #         self.out_mem = org_info['org'].forward(self.out_mem)
    #         org_info['age'] += 1
    #         if org_info['age'] > 500 and org_info['infra'] < 1:
    #             genomes_to_remove.append(genome_key)

    #     for genome_key in genomes_to_remove:
    #         self.population.pop(genome_key)

    #     if len(self.population.keys()) > self.max_size:
    #         # Calculate how many genomes to remove
    #         num_to_remove = len(self.population) - self.max_size
    #         # Sort genomes by infra value (ascending order) and select the ones to remove
    #         genomes_to_remove = sorted(self.population.items(), key=lambda x: x[1]['infra'])[:num_to_remove]
    #         # Remove the selected genomes
    #         for genome_key, _ in genomes_to_remove:
    #             self.population.pop(genome_key)

    #     if len(self.population) < self.min_size:
    #         self.init_random_pop(self.min_size - len(self.population))
        
    #     self.substrate.mem[0, self.act_chinds] = self.out_mem
    #     self.apply_physics()
    #     self.time_step += 1


    # def sexual_reproduction(self, merging_cell_coords, incoming_genome_matrix):
    #     inds = self.substrate.ti_indices[None]
    #     if len(merging_cell_coords[0]) == 0:
    #         return
    #     for i in range(merging_cell_coords[0].shape[0]):
    #         x, y = merging_cell_coords[0][i].item(), merging_cell_coords[1][i].item()
    #         old_genome_key = self.substrate.mem[0, inds.genome, x, y].item()
    #         incoming_genome_key = incoming_genome_matrix[x, y].item()
    #         incoming_org = self.population[incoming_genome_key]['org']
    #         if (old_genome_key == -1 or
    #             old_genome_key == incoming_genome_matrix[x, y]
    #             or old_genome_key not in self.population):
    #             self.substrate.mem[0, inds.genome, x, y] = incoming_genome_key
    #             continue
    #         old_org = self.population[old_genome_key]['org']
    #         child_genome = neat.DefaultGenome(str(self.next_free_genome_key))
    #         child_genome.configure_crossover(old_org.genome, incoming_org.genome, self.neat_config)
    #         child_organism = self.create_organism(genome_key=self.next_free_genome_key, genome=child_genome)
    #         self.population[self.next_free_genome_key] = {"org": child_organism, "infra": 0.1, "age": 0}
    #         self.substrate.mem[0, inds.genome, x, y] = self.next_free_genome_key
    #         self.next_free_genome_key += 1




    # def get_genome_infra_sum(self, genome_key):
    #     inds = self.substrate.ti_indices[None]
    #     infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
    #     return infra_sum


    # def update_population_infra_sum(self):
    #     for genome_key in self.population.keys():
    #         infra_sum = self.get_genome_infra_sum(genome_key)
    #         self.population[genome_key]["infra"] = infra_sum
    #         self.population[genome_key]["org"].fitness = infra_sum



    # def get_random_coords_of_genome(self, genome_key, n_coords=1):
    #     inds = self.substrate.ti_indices[None]
    #     genome_coords = torch.where(self.substrate.mem[0, inds.genome] == genome_key)
    #     if genome_coords[0].shape[0] == 0:  # Check if there are no matching coordinates
    #         return []  # Return an empty list or handle this case as needed
    #     random_indices = torch.randint(0, genome_coords[0].shape[0], (n_coords,))
    #     x_coords = genome_coords[0][random_indices]
    #     y_coords = genome_coords[1][random_indices]
    #     coords = torch.stack((x_coords, y_coords), dim=1)
    #     return coords.tolist()


    # def get_random_genome_keys(self, n_genomes):
    #     inds = self.substrate.ti_indices[None]
    #     infras = torch.tensor([org_info['infra'] for org_info in self.population.values()], dtype=torch.float32)
    #     infra_sum = infras.sum()
    #     if infra_sum != 0:
    #         infra_probs = infras / infra_sum
    #     else:
    #         # Create a uniform distribution if infra_sum is 0
    #         infra_probs = torch.ones_like(infras) / len(infras)
    #     selected_index = torch.multinomial(infra_probs, n_genomes, replacement=True)
    #     selected_genomes = [list(self.population.keys())[i] for i in selected_index]
    #     return selected_genomes