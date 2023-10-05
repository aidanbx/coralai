# %% 
from matplotlib import animation
import numpy as np
import noise
from scipy.stats import uniform
from scipy.stats import levy_stable
from scipy import signal
import torch
import yaml
import random
import matplotlib.pyplot as plt

import importlib

def discretize_levy_dust(dust: np.array, channel: np.array) -> np.array:
    """Discretize a levy dust cloud into a grid of shape shape,
    such that each position in the grid is the number of points in the cloud that fall in that position

    Returns:
        np.array: Grid of shape shape representing the density of points in the dust cloud
    """
    dust = np.array(dust, dtype=np.int64)
    points, density = np.unique(dust, axis=1, return_counts=True)
    channel[points[0,:], points[1,:]] = density


def levy_dust(shape: tuple, points: int, alpha: float, beta: float) -> np.array:
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi) 

    step_length = abs(levy_stable.rvs(alpha, beta, size=points))

    x = np.cumsum(step_length * np.cos(angle)) % (shape[0])
    y = np.cumsum(step_length * np.sin(angle)) % (shape[1])

    return np.array([x, y])


def populate_obstacle(channel: np.array, config=None):
    if config is None:
        config =  {
            "environment": {
                "obstacle_generation": {
                    "method": "simplex_noise",
                    "empty_threshold": [0.3, 0.4],
                    "full_threshold": [0.6, 0.7],
                    "frequency": [4.0, 30.0],
                    "octaves": [1, 10],
                    "persistence": [0.2, 1.0],
                    "lacunarity": [1.5, 4.0]
                }
            }
        }
    obstacle_params = config["environment"]["obstacle_generation"]
    
    empty_threshold = obstacle_params.get("empty_threshold")
    full_threshold = obstacle_params.get("full_threshold")
    frequency_range = obstacle_params.get("frequency")
    octaves_range = obstacle_params.get("octaves")
    persistence_range = obstacle_params.get("persistence")
    lacunarity_range = obstacle_params.get("lacunarity")

    empty_threshold = random.uniform(*empty_threshold)
    full_threshold = random.uniform(*full_threshold)
    frequency = random.uniform(*frequency_range)
    octaves = random.randint(*octaves_range)
    persistence = random.uniform(*persistence_range)
    lacunarity = random.uniform(*lacunarity_range)

    # Generate random offsets for x and y
    x_offset = random.randint(0, 10000)
    y_offset = random.randint(0, 10000)

    for x in range(channel.shape[0]):
        for y in range(channel.shape[1]):
            value = noise.pnoise2((x + x_offset) / frequency, (y + y_offset) / frequency, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            channel[x,y] = value
    
    # Normalize the channel
    channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    channel = np.where(channel > full_threshold, 1, channel)
    channel = np.where(channel < empty_threshold, 0, channel)
    return channel