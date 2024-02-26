import torch
import os
import neat
import taichi as ti
import uuid

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.recurrent_net import RecurrentNet
from ...dynamics.organism import Organism

@ti.data_oriented
class MinimalOrganism(Organism):
    def __init__(self, batch_size, n_sensors, n_actuators,
                 sensor_names, actuator_names, torch_device):
        super().__init__(n_sensors, n_actuators)

        self.batch_size = batch_size
        self.sensor_names = sensor_names
        self.actuator_names = actuator_names
        self.torch_device = torch_device

        config_filename = "minimal_neat.config"
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_filename)

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
        

        self.genome = self.gen_random_genome()
        self.convert_to_torch()


    def gen_random_genome(self):
        # Create a new genome with a unique UUID
        genome_id = uuid.uuid4()
        genome = neat.DefaultGenome(str(genome_id))
        
        # Initialize the new genome with the configuration settings
        genome.configure_new(self.config.genome_config)
        
        return genome
    

    def convert_to_torch(self):
        self.net = RecurrentNet.create(self.genome, self.config,  batch_size=self.batch_size, dtype=torch.float32, device=self.torch_device)

    def forward(self, x):
        # Turn x from shape (1,1,400,400) to (160000, 1)
        x = torch.tensor(x, device=self.torch_device, dtype=torch.float32)
        x = x.flatten()
        x = x.unsqueeze(1)
        x = self.net.activate(x)
        
        x = x.squeeze(1)
        x = x.view(1, 1, 400, 400)
        return x


    def mutate(self, weight_mutate_rate, weight_mutate_power, bias_mutate_rate):
        # Example: Adjusting the NEAT config directly (simplified for demonstration)
        self.config.genome_config.weight_mutate_rate = weight_mutate_rate
        self.config.genome_config.weight_mutate_power = weight_mutate_power
        self.config.genome_config.bias_mutate_rate = bias_mutate_rate
        
        # Apply mutations using the updated config
        self.genome.mutate(self.config.genome_config)
        self.convert_to_torch()

