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
import src.pcg as pcg
importlib.reload(pcg)

class Resource:
    # a resource as an id (incremental), a min and max,
    # regeneration and dispersal functions of time,
    # a distribution function (levy dust usually)

    # a resource exists in the resource map as an id on a given cell along with a value
    # the id determines how to update it using the respective functions in this class

    def __init__(self, id, regen_func, metadata=None, dispersal_func=None):
        self.id = id
        self.regen_func = regen_func
        default_metadata = {
            'id': id
            }
        if metadata is None:
            metadata = {}
        metadata.update(default_metadata)
        self.metadata = metadata
        self.dispersal_func = dispersal_func

    # def update(self, time, resource_map, port):
    #     port[resource_map == self.id] = port[resource_map == self.id] + self.regen_func(time), self.min, self.max)

def generate_graphics(resources, resource_map, port, min_val, max_val, num_iters=1000):
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
