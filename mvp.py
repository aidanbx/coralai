from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import torch
import importlib
import numpy as np
import matplotlib.pyplot as plt
import src.generate_env as gen_env
importlib.reload(gen_env)

import src.physics as physics
importlib.reload(physics)

import src.EINCASMConfig as EINCASMConfig
importlib.reload(EINCASMConfig)

import src.Resource as Resource
importlib.reload(Resource)

cfg = EINCASMConfig.Config('config.yaml')

VERBOSE = True

num_scents = cfg.num_resource_types
perception_indecies = {
    "scent": [i for i in range(0, num_scents)],
    "capital": [i for i in range(num_scents, num_scents + cfg.kernel.shape[0])],
    "muscles": [i for i in range(num_scents + cfg.kernel.shape[0], num_scents + cfg.kernel.shape[0]*2)],
    "communication": [i for i in range(num_scents + cfg.kernel.shape[0]*2, num_scents + cfg.kernel.shape[0]*2 + cfg.num_communication_channels)]
}

actuator_indecies = {
    "growth_activation": [i for i in range(cfg.kernel.shape[0])], # Flow muscles for kernel + mine + resource muscles
    "communication": [i for i in range(cfg.kernel.shape[0], cfg.kernel.shape[0]+cfg.num_communication_channels)],
    "flow_activation": cfg.kernel.shape[0]+cfg.num_communication_channels,
    "mine_activation": cfg.kernel.shape[0]+cfg.num_communication_channels+1,
    "resource_activation": cfg.kernel.shape[0]+cfg.num_communication_channels+2
}

if VERBOSE:
    print(perception_indecies, actuator_indecies)

obstacles = np.zeros(cfg.world_shape)
gen_env.populate_obstacle(obstacles)
obstacles = torch.from_numpy(obstacles).float()

resource_map = torch.zeros(cfg.world_shape, dtype=torch.int8)
port = torch.zeros(cfg.world_shape, dtype=torch.float32)

resources = []
num_resources = 3
min_val = -10
max_val = 10
# repeat_intervals = []
for i in range(1,num_resources+1):
    regen_func, freqs, amps, start_periods = Resource.gen_random_signal_func()
    # repeat_interval = 2 * np.pi / np.gcd.reduce(freqs.numpy())
    # repeat_intervals.append(repeat_interval)
    resources.append(Resource.Resource(i, min_val, max_val, regen_func))
    resource_map, port = resources[i-1].populate_map(resource_map, port)
