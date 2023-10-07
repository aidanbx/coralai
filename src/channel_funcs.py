import torch
import numpy as np
import importlib
import src.pcg as pcg
importlib.reload(pcg)
import src.Resource as Resource
importlib.reload(Resource)
import random


def init_obstacles_perlin(shape: tuple, metadata: dict):
    # TODO: update to batches? 
    empty_threshold = metadata.get("empty_thresh", 0.4)
    full_threshold = metadata.get("full_thresh", 0.6)
    frequency = metadata.get("frequency", 15)
    octaves = metadata.get("octaves", 9)
    persistence = metadata.get("persistence", 0.6)
    lacunarity = metadata.get("lacunarity", 1.5)

    x_offset = random.randint(0, 10000)
    y_offset = random.randint(0, 10000)

    obstacles = pcg.perlin2d(shape[0], shape[1], frequency, octaves, persistence,
                    lacunarity, x_offset, y_offset, normalized=True)
    torch.where(obstacles > full_threshold, torch.tensor(1), obstacles, out=obstacles)
    torch.where(obstacles < empty_threshold,torch.tensor(0), obstacles, out=obstacles)

    return obstacles, {"empty_threshold": empty_threshold,
                        "full_threshold": full_threshold,
                        "frequency": frequency,
                        "octaves": octaves,
                        "persistence": persistence,
                        "lacunarity": lacunarity,
                        "x_offset": x_offset,
                        "y_offset": y_offset}


def init_resources_levy(shape: tuple, metadata: dict):
    port_id_map = torch.zeros(shape, dtype=torch.int8)
    port_sizes = torch.zeros(shape)
    resources = []

    for port_id in range(1,metadata["num_resources"]+1):
        # regen_func, freqs, amps, start_periods
        signal_info = pcg.random_signal(
        min_amp = metadata["min_regen_amp"], max_amp=metadata["max_regen_amp"])
        regen_func = signal_info[0]

        resource = Resource.Resource(port_id, regen_func)
        resources.append(resource)
        alpha = random.uniform(*metadata["alpha_range"])
        beta = random.uniform(*metadata["beta_range"])
        num_sites = random.randint(*metadata["num_sites_range"])

        dust = pcg.levy_dust(shape, num_sites, alpha, beta)
        dust = pcg.discretize_levy_dust(shape, dust)
        port_id_map[dust > 0] = port_id
        port_sizes += dust

        resource.metadata.update({'regen_signal_info': {
                                        "frequencies": signal_info[1].tolist(),
                                        "amplitudes": signal_info[2].tolist(),
                                        "start_periods": signal_info[3].tolist()
                                        },
                                    'alpha': alpha,
                                    'beta': beta,
                                    'num_sites': num_sites,
                                    'object': resource})
    port_metadata = {
        'port_id_map': port_id_map,
        'port_sizes': port_sizes,
        'resources': resources
    }
    for resource in resources:
        port_metadata[f"resource_{resource.id}_init_info"] = resource.metadata
    return torch.empty(shape), port_metadata


def init_live_channels(self):
    self.muscle_radii = torch.zeros((self.cfg.num_muscles, *self.cfg.world_shape), device=self.cfg.device, dtype=self.cfg.float_dtype)
    self.capital = torch.zeros(self.cfg.world_shape, device=self.cfg.device, dtype=self.cfg.float_dtype)
    self.waste = torch.zeros(self.cfg.world_shape, device=self.cfg.device, dtype=self.cfg.float_dtype)
    return self.muscle_radii, self.capital, self.waste

def init_costs(self):
    # These could all be wxh tensors too, if you want weather to act on efficiencies or other things...
    self.port_cost = torch.tensor(0.01, device=self.cfg.device, dtype=self.cfg.float_dtype)
    self.mining_cost = self.port_cost
    self.growth_cost = self.port_cost
    self.flow_cost = self.port_cost
    return self.port_cost, self.mining_cost, self.growth_cost, self.flow_cost
