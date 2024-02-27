import os
import torch
import neat
import torch.nn as nn
import taichi as ti
import uuid

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.recurrent_net import RecurrentNet
from ...dynamics.nn_lib import ch_norm
from ...dynamics.organism import Organism

@ti.data_oriented
class MinimalOrganism(Organism):
    def __init__(self, substrate, sensors, n_actuators, torch_device):
        super().__init__(substrate, sensors, n_actuators)

        self.torch_device = torch_device

        config_filename = "nca_neat.config"
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_filename)

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
        

        self.genome = self.gen_random_genome()


    def gen_random_genome(self):
        # Create a new genome with a unique UUID
        genome_id = uuid.uuid4()
        genome = neat.DefaultGenome(str(genome_id))
        
        # Initialize the new genome with the configuration settings
        genome.configure_new(self.config.genome_config)
        
        return genome


    def forward(self, mem):
        genome_key = 0
        genome_map = torch.zeros(mem.shape[2], mem.shape[3], dtype=torch.int32, device=self.torch_device)
        num_cells_of_genome = torch.sum(genome_map.eq(genome_key))
        matches = genome_map.eq(genome_key)
        coords = torch.where(matches)
        sensors = torch.zeros(num_cells_of_genome, 9, dtype=torch.float32, device=self.torch_device)
        x_coords, y_coords = coords[0].contiguous(), coords[1].contiguous()
        self.sense_to(mem, sensors, x_coords, y_coords)

        net = RecurrentNet.create(self.genome, self.config, batch_size=num_cells_of_genome, dtype=torch.float32, device=self.torch_device)
        actuators = net.activate(sensors)
        
        mem = torch.zeros_like(self.substrate.mem)
        self.act_on(actuators, mem, x_coords, y_coords)
        # x = x.reshape(1, 1, 256, 256)
        mem = ch_norm(mem)
        mem = nn.ReLU()(mem)
        mem = torch.sigmoid(mem)
        return mem


    def mutate(self, rate):
        self.genome.mutate(self.config.genome_config)