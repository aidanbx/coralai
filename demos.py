import torch
import numpy as np

import importlib
import src.Resource as Resource
importlib.reload(Resource)
import src.pcg as pcg
importlib.reload(pcg)

def resource_weather_demo():
    resource_map = torch.zeros((100,100), dtype=torch.int8)
    port = torch.zeros((100,100), dtype=torch.float32)

    resources = []
    num_resources = 3
    min_val = -10
    max_val = 10

    regen_func, freqs, amps, start_periods = pcg.random_signal_func(num_components=6, min_freq=np.pi/10, max_freq=np.pi*8, min_amp=0.5, max_amp=0.5)
    resources.append(Resource.Resource(1, min_val, max_val, regen_func))
    resource_map, port = resources[0].populate_map(resource_map, port)

    regen_func, freqs, amps, start_periods = pcg.random_signal_func(num_components=2, min_freq=np.pi/10, max_freq=np.pi)
    resources.append(Resource.Resource(2, min_val, max_val, regen_func))
    resource_map, port = resources[1].populate_map(resource_map, port)

    regen_func, freqs, amps, start_periods = pcg.random_signal_func()
    resources.append(Resource.Resource(3, min_val, max_val, regen_func))
    resource_map, port = resources[2].populate_map(resource_map, port)

    Resource.generate_graphics(resources, resource_map, port, min_val, max_val, 1000)

if __name__ == "__main__":
    resource_weather_demo()
