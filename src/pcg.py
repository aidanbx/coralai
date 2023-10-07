# %% 
import numpy as np
import noise
from scipy.stats import uniform
from scipy.stats import levy_stable
from scipy import signal
import torch
import random

def perlin2d(width, height, frequency=10.0, octaves=4, persistence=0.6,
                       lacunarity=3.0, x_offset=0, y_offset=0, channel=None, normalized=True):
    if channel is None:
        channel = torch.zeros((width, height))
    
    for x in range(width):
        for y in range(height):
            value = noise.pnoise2((x + x_offset) / frequency, (y + y_offset) / frequency, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            channel[x,y] = value
    
    if normalized:
        channel = (channel - channel.min()) / (channel.max() - channel.min())

    return channel

def discretize_levy_dust(shape: tuple, dust: np.array) -> np.array:
    """Discretize a levy dust cloud into a grid of shape shape,
    such that each position in the grid is the number of points in the cloud that fall in that position

    Returns:
        np.array: Grid of shape shape representing the density of points in the dust cloud
    """
    dust = np.array(dust, dtype=np.int64)
    points, density = np.unique(dust, axis=1, return_counts=True)
    out = np.zeros(shape)
    out[points[0,:], points[1,:]] = torch.from_numpy(density).float()
    return out

def levy_dust(shape: tuple, points: int, alpha: float, beta: float) -> np.array:
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi) 

    # Ensure alpha and beta are within the valid range for the levy_stable distribution
    alpha = max(min(alpha, 2), 0.1)
    beta = max(min(beta, 1), -1)

    step_length = abs(levy_stable.rvs(alpha, beta, size=points))

    x = np.cumsum(step_length * np.cos(angle)) % (shape[0])
    y = np.cumsum(step_length * np.sin(angle)) % (shape[1])

    return np.array([x, y])

def random_signal(num_components=2,
                    min_freq=np.pi/10, max_freq=np.pi*2, min_amp=0.1, max_amp=0.4,
                    min_start_period=0, max_start_period=np.pi*2):

    freqs = (max_freq - min_freq) * torch.rand(num_components) + min_freq
    # rots = (max_rot - min_rot) * torch.rand(num_signals).to(device) + min_rot
    amps = (max_amp - min_amp) * torch.rand(num_components) + min_amp
    start_periods = (max_start_period - min_start_period) * torch.rand(num_components) + min_start_period

    return lambda t: sum([amps[i] * torch.sin(freqs[i] * t + start_periods[i]) for i in range(num_components)]), freqs, amps, start_periods
