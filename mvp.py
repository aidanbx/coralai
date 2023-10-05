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
num_resources = 4
min_val = -10
max_val = 10
# repeat_intervals = []
for i in range(1,num_resources+1):
    regen_func, freqs, amps, start_periods = Resource.gen_random_signal_func()
    # repeat_interval = 2 * np.pi / np.gcd.reduce(freqs.numpy())
    # repeat_intervals.append(repeat_interval)
    resources.append(Resource.Resource(i, min_val, max_val, regen_func))
    resource_map, port = resources[i-1].populate_map(resource_map, port)


# if VERBOSE:
#     num_iters = 100

#     fig, ax = plt.subplots(1,2, figsize=(10,5))

#     def update(i):
#         for resource in resources:
#             resource.update(i, resource_map, port)
#         ax[0].cla()
#         ax[1].cla()
#         ax[0].imshow(resource_map, cmap="tab20b")
#         ax[1].imshow(port, cmap="PiYG")

#     ani = FuncAnimation(fig, update, frames=num_iters, repeat=False)
#     plt.show()


def generate_graphics(resources, num_iters=1000, shift_signal=True):
    # Setting up the visualization using GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.25])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    colors = plt.get_cmap("tab20b")(np.linspace(0, 1, len(resources)))

    def update(i):
        # Top-left: Resource ID map
        ax1.clear()
        ax1.imshow(resource_map, cmap="tab20b")
        
        # Top-right: Resource values
        ax2.clear()
        ax2.imshow(port, cmap="PiYG")

        # Bottom: Regeneration functions
        ax3.clear()
        for idx, resource in enumerate(resources):
            t = torch.linspace(0, 2*np.pi, 1000)
            if shift_signal:
                y = resource.regen_func(t + i*0.01).numpy()  # Shift signals over time
            else:
                y = resource.regen_func(t).numpy()
            ax3.plot(t, y, color=colors[idx], label=f"Resource {idx+1}")

        ax3.axvline(x=i*0.01 % (2*np.pi), color='black', linestyle='--')
        ax3.legend(loc="upper right")

        # Update resources
        for resource in resources:
            resource.update(i, resource_map, port)

    ani = FuncAnimation(fig, update, frames=num_iters, interval=20, repeat=True)
    plt.tight_layout()
    plt.show()

# For the shifting signal version
generate_graphics(resources, shift_signal=True)
# For the static signals version
generate_graphics(resources, shift_signal=False)

