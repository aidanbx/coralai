import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.stats import uniform
from scipy.stats import levy_stable
import importlib

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

def gen_random_signal_func(num_components=2,
                    min_freq=np.pi/10, max_freq=np.pi*2, min_amp=0.1, max_amp=0.4,
                    min_start_period=0, max_start_period=np.pi*2):

    freqs = (max_freq - min_freq) * torch.rand(num_components) + min_freq
    # rots = (max_rot - min_rot) * torch.rand(num_signals).to(device) + min_rot
    amps = (max_amp - min_amp) * torch.rand(num_components) + min_amp
    start_periods = (max_start_period - min_start_period) * torch.rand(num_components) + min_start_period

    return lambda t: sum([amps[i] * torch.sin(freqs[i] * t + start_periods[i]) for i in range(num_components)]), freqs, amps, start_periods

class Resource:
    # a resource as an id (incremental), a min and max,
    # regeneration and dispersal functions of time,
    # a distribution function (levy dust usually)

    # a resource exists in the resource map as an id on a given cell along with a value
    # the id determines how to update it using the respective functions in this class

    def __init__(self, id, min, max, regen_func, dispersal_func=None):
        self.id = id
        self.min = min
        self.max = max
        self.regen_func = regen_func
        self.dispersal_func = dispersal_func
        
        self.alpha = random.uniform(0.4, 0.9)
        self.beta = random.uniform(0.8, 1.2)
        self.num_sites = random.randint(50, 100)


    def update_distribution_params(self, alpha, beta, num_sites):
        self.alpha = alpha
        self.beta = beta
        self.num_sites = num_sites
    

    def populate_map(self, resource_map, port):
        dust = levy_dust(port.shape, self.num_sites, self.alpha, self.beta)
        dust = discretize_levy_dust(port.shape, dust)
        # port += dust
        resource_map[dust > 0] = self.id

        return resource_map, port

    def update(self, time, resource_map, port):
        port[resource_map == self.id] = torch.clamp(port[resource_map == self.id] + self.regen_func(time), self.min, self.max)

def generate_graphics(resources, resource_map, port, min_val=-10, max_val=10, num_iters=1000):
    # Colors
    resource_colors = ['red', 'green', 'blue']

    # Setting up the visualization using GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.25])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    def update(i):
        # Top-left: Resource ID map
        ax1.clear()
        ax1.set_title("Resource ID map")
        cmap = mcolors.ListedColormap(['white'] + resource_colors)
        ax1.imshow(resource_map, cmap=cmap)

        # Top-right: Resource values
        ax2.clear()
        ax2.set_title("Resource Values (Yl Good)")
        midmap = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        ax2.imshow(port, cmap="viridis", norm=midmap)
                     
        ax3.clear()
        ax3.set_title("Regeneration functions")
        for idx, resource in enumerate(resources):
            t = torch.linspace(0, 2*np.pi, 1000)
            y = resource.regen_func(t+i-np.pi/2).numpy()
            ax3.plot(t+i, y, color=resource_colors[idx], label=f"Resource {idx+1}")

        ax3.axvline(x=i + np.pi/2, color='black', linestyle='--')  # Static vertical line
        ax3.axvspan(i, i + np.pi/2, facecolor='gray', alpha=0.2)  # Highlight indicating past
        ax3.legend(loc="upper right")

        # Update resources
        for resource in resources:
            resource.update(i, resource_map, port)

    ani = FuncAnimation(fig, update, frames=np.linspace(0, np.pi*8, num_iters), interval=50, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    resource_map = torch.zeros((100,100), dtype=torch.int8)
    port = torch.zeros((100,100), dtype=torch.float32)

    resources = []
    num_resources = 3
    min_val = -10
    max_val = 10

    regen_func, freqs, amps, start_periods = gen_random_signal_func(num_components=6, min_freq=np.pi/10, max_freq=np.pi*8, min_amp=0.5, max_amp=0.5)
    resources.append(Resource(1, min_val, max_val, regen_func))
    resource_map, port = resources[0].populate_map(resource_map, port)

    regen_func, freqs, amps, start_periods = gen_random_signal_func(num_components=2, min_freq=np.pi/10, max_freq=np.pi)
    resources.append(Resource(2, min_val, max_val, regen_func))
    resource_map, port = resources[1].populate_map(resource_map, port)

    regen_func, freqs, amps, start_periods = gen_random_signal_func()
    resources.append(Resource(3, min_val, max_val, regen_func))
    resource_map, port = resources[2].populate_map(resource_map, port)
    
    generate_graphics(resources, resource_map, port, min_val, max_val, 1000)