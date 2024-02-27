import os
import copy
import torch
import neat
import torch.nn as nn
import taichi as ti
import uuid

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.recurrent_net import RecurrentNet
from ..dynamics.nn_lib import ch_norm
from .organism import Organism

@ti.data_oriented
class EvolvableOrganism(Organism):
    def __init__(self, neat_config, substrate, sensors, n_actuators, torch_device):
        super().__init__(substrate, sensors, n_actuators)

        self.torch_device = torch_device

        self.neat_config = neat_config
        
        self.genome = None
        self.genome_key = None
        self.net = None


    def set_genome(self, genome_key, genome=None):
        self.genome_key = genome_key
        if genome is None:
            self.genome = self.gen_random_genome()
        else:
            self.genome = genome
        self.net = RecurrentNet.create(self.genome, self.neat_config, batch_size=self.substrate.w*self.substrate.h, dtype=torch.float32, device=self.torch_device)

    
    def gen_random_genome(self):
        # Create a new genome with a unique UUID
        genome_id = uuid.uuid4()
        genome = neat.DefaultGenome(str(genome_id))
        
        # Initialize the new genome with the configuration settings
        genome.configure_new(self.neat_config.genome_config)
        
        return genome


    def get_genome_coords(self, genome_map):
        matches = genome_map.eq(self.genome_key)
        coords = torch.where(matches)
        x_coords, y_coords = coords[0].contiguous(), coords[1].contiguous()

        return x_coords, y_coords
    

    @ti.kernel
    def sense_to(self, mem: ti.types.ndarray(), sensors: ti.types.ndarray(),
                 x_coords: ti.types.ndarray(), y_coords: ti.types.ndarray()):
        for i in ti.ndrange(x_coords.shape[0]):
            x = x_coords[i]
            y = y_coords[i]
            neigh_num = 0
            for off_x, off_y in ti.ndrange((-1, 2), (-1, 2)):
                sensors[i, neigh_num] = mem[0, 0, (x + off_x) % mem.shape[2], (y+off_y) % mem.shape[3]]
                neigh_num += 1


    @ti.kernel
    def act_on(self, actuators: ti.types.ndarray(), mem: ti.types.ndarray(),
               x_coords: ti.types.ndarray(), y_coords: ti.types.ndarray()):
        for i in ti.ndrange(x_coords.shape[0]):
            mem[0, 0, x_coords[i], y_coords[i]] = actuators[i,0]


    def forward(self, mem, genome_map):
        print("\n-----------------\n")
        print("Mem Stats:")
        print(f"Max Value: {mem.max().item()}")
        print(f"Min Value: {mem.min().item()}")
        print(f"Mean Value: {mem.mean().item()}")
        print(f"Standard Deviation: {mem.std().item()}")
        # add random noise to mem
        mem = mem + torch.randn_like(mem) * 0.1

        x_coords, y_coords = self.get_genome_coords(genome_map)
        num_cells_of_genome = x_coords.shape[0]
        sensors = torch.zeros(num_cells_of_genome, 9, dtype=torch.float32, device=self.torch_device)
        self.sense_to(mem, sensors, x_coords, y_coords)

        actuators = self.net.activate(sensors)
        self.act_on(actuators, mem, x_coords, y_coords)
        mem = nn.ReLU()(mem)
        mem = ch_norm(mem)
        mem = torch.sigmoid(mem)
        print("Actuator Stats:")
        print(f"Max Value: {mem.max().item()}")
        print(f"Min Value: {mem.min().item()}")
        print(f"Mean Value: {mem.mean().item()}")
        print(f"Standard Deviation: {mem.std().item()}")
        # x = x.reshape(1, 1, 256, 256)
        
        return mem


    def mutate(self, rate):
        cloned_genome = copy.deepcopy(self.genome)
        cloned_genome.mutate(self.neat_config.genome_config)
        return cloned_genome