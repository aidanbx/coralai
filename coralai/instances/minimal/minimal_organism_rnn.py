import os
import torch
import neat
import torch.nn as nn
import taichi as ti
import uuid

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.recurrent_net import RecurrentNet
from ...dynamics.nn_lib import ch_norm
from ...evolution.evolvable_organism import EvolvableOrganism

@ti.data_oriented
class MinimalOrganism(EvolvableOrganism):
    def __init__(self, neat_config, substrate, sensors, n_actuators, torch_device):
        super().__init__(neat_config, substrate, sensors, n_actuators, torch_device)
