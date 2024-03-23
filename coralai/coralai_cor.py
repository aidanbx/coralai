import os

import torch
import neat

from coralai.substrate import Substrate
from .reportable import Reportable

class CoralaiCor(Reportable):
    def __init__(self, config_path, channels, shape, kernel,
                 sense_chs, act_chs, torch_device, metadata=None):
        super().__init__(metadata)
        self.config_path = config_path
        self.substrate = Substrate(shape, torch.float32, torch_device, channels)
        self.substrate.malloc()

        self.inds = self.substrate.ti_indices[None]
        self.kernel = torch.tensor(kernel, device=torch_device)
        self.sense_chs = sense_chs
        self.sense_chinds = self.substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = self.substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)

        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config_path)
        self.torch_device = torch_device
    
    def report(self):
        pass

