from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import animation 
from matplotlib import gridspec
import matplotlib.colors as mcolors
import torch
import datetime
import os
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
cfg.update_device('cpu')

VERBOSE = True

obstacles = np.zeros(cfg.world_shape)
obstacles = gen_env.populate_obstacle(obstacles)
obstacles = torch.from_numpy(obstacles).float().to(cfg.device)
assert obstacles.min() >= 0, "Obstacles cannot be negative (before update)"

resource_map = torch.zeros(cfg.world_shape, device=cfg.device, dtype=torch.int8)
port = torch.zeros(cfg.world_shape, device=cfg.device, dtype=cfg.float_dtype)

resources = []
num_resources = 3
min_val = -1
max_val = 10
# repeat_intervals = []
for i in range(1,num_resources+1):
    regen_func, freqs, amps, start_periods = Resource.gen_random_signal_func(min_amp=0.5, max_amp=2)
    # repeat_interval = 2 * np.pi / np.gcd.reduce(freqs.numpy())
    # repeat_intervals.append(repeat_interval)
    resources.append(Resource.Resource(i, min_val, max_val, regen_func))
    resource_map, port = resources[i-1].populate_map(resource_map, port)

muscle_radii = torch.zeros((cfg.num_muscles, *cfg.world_shape), device=cfg.device, dtype=cfg.float_dtype)
capital = torch.zeros(cfg.world_shape, device=cfg.device, dtype=cfg.float_dtype)
waste = torch.zeros(cfg.world_shape, device=cfg.device, dtype=cfg.float_dtype)

# These could all be wxh tensors too, if you want weather to act on efficiencies or other things...
port_cost = torch.tensor(0.01, device=cfg.device, dtype=cfg.float_dtype)
mining_cost = port_cost
growth_cost = port_cost
flow_cost = port_cost

num_scents = cfg.num_resource_types
num_com = cfg.num_communication_channels * cfg.kernel.shape[0]

perception_indecies = {
    "scent":         [i                                         for i in range(0, num_scents)],
    "capital":       [i + num_scents                            for i in range(cfg.kernel.shape[0])],
    "muscles":       [i + num_scents + cfg.kernel.shape[0]      for i in range(cfg.kernel.shape[0])],
    "communication": [i + num_scents + cfg.kernel.shape[0]*2    for i in range(num_com)]
}
total_perception_channels = num_scents + cfg.kernel.shape[0]*2 + num_com

actuator_indecies = {
    "flow_act":  0,
    "mine_act":  1,
    "port_act":  2,
    "muscle_radii_delta":   [i+3                    for i in range(cfg.num_muscles)],
    "communication":        [i+3+cfg.num_muscles    for i in range(cfg.num_communication_channels)]
}
total_actuator_channels = cfg.kernel.shape[0] + 3 + cfg.num_communication_channels

def random_agent(perception):
    actuators = torch.rand(total_actuator_channels)*2-1
    return actuators

def grow(actuators):
    muscle_radii_delta = actuators[actuator_indecies["muscle_radii_delta"]]
    physics.grow_muscle_csa(cfg, muscle_radii, muscle_radii_delta, capital, growth_cost)

def flow(actuators):
    flow_act = actuators[actuator_indecies["flow_act"]]
    physics.activate_flow_muscles(cfg, capital, waste, muscle_radii[:-2], flow_act, flow_cost, obstacles)

def eat(actuators):
    port_act = actuators[actuator_indecies["port_act"]]
    physics.activate_port_muscles(cfg, capital, port, muscle_radii[-2], port_act, port_cost)

def dig(actuators):
    mine_act = actuators[actuator_indecies["mine_act"]]
    physics.activate_mine_muscles(muscle_radii[-1], mine_act, capital, obstacles, waste, mining_cost)

update_functions = [grow, flow, eat, dig]



timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
base_folder = "runs/"
folder = os.path.join(base_folder, timestamp)

# Ensure the base folder exists
if not os.path.exists(base_folder):
    os.mkdir(base_folder)

# Handle potential collisions
counter = 0
while os.path.exists(folder):
    counter += 1
    folder = os.path.join(base_folder, f"{timestamp}-{counter}")

os.mkdir(folder)

def repopulate_capital(num_sites):
    # Find indices of port channel with highest value
    values, flat_indices = port.view(-1).topk(num_sites)
    indices = np.column_stack(np.unravel_index(flat_indices.numpy(), port.shape))

    capital[indices[:,0], indices[:,1]] = 1000

repopulate_capital(100)

if VERBOSE:
    save_path = os.path.join(folder, f"init.pt")
    torch.save({
        'muscle_radii': muscle_radii,
        'capital': capital,
        'waste': waste,
        'port': port,
        'resource_map': resource_map,
        'obstacles': obstacles,
    }, save_path)

    data = []

def update(period):
    for resource in resources:
        resource.update(period, resource_map, port)

    actuators = torch.randn((total_actuator_channels, *cfg.world_shape), device=cfg.device, dtype=cfg.float_dtype) * 2-1
    actuators = torch.where(capital>0.1, actuators, torch.zeros_like(actuators))
    random_update_order = np.random.permutation(update_functions)
    for update_function in random_update_order:
        update_function(actuators)
    if capital.sum() < 20:
        repopulate_capital(200)

    if VERBOSE:
        save_path = os.path.join(folder, f"{period}.pt")
        dic = {
            'muscle_radii': muscle_radii,
            'capital': capital,
            'waste': waste,
            'port': port,
            'resource_map': resource_map,
            'obstacles': obstacles,
            'actuators': actuators
        }
        data.append(dic)
        torch.save(dic, save_path)
        
periods = np.linspace(0,np.pi*2,100)
for period in periods:
    update(period)

print("Done")

def animate():
    # Colors
    resource_colors = ['red', 'green', 'blue']

    # Setting up the visualization using GridSpec
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,0.25])
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tm = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[0, 2])
    ax_ml = fig.add_subplot(gs[1, 0])
    ax_mm = fig.add_subplot(gs[1, 1])
    ax_mr = fig.add_subplot(gs[1, 2])
    ax_bl = fig.add_subplot(gs[2, 0])
    ax_b = fig.add_subplot(gs[2, 1:])

    ax_rmap = ax_mm
    ax_port = ax_mr
    ax_regen = ax_b
    ax_waste = ax_ml
    ax_capital = ax_tl
    ax_activation = ax_tm
    ax_obstacle = ax_tr

    def update_ani(i):
        period = periods[i]
        # Top Middle: Resource ID map
        ax_rmap.clear()
        ax_rmap.set_title('Resource Map')
        cmap = mcolors.ListedColormap(['white'] + resource_colors)
        ax_rmap.imshow(resource_map, cmap=cmap)

        # Middle Right: Resource Values
        ax_port.clear()
        ax_port.set_title("Resource Values (Yl Good)")
        midmap = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        ax_port.imshow(data[i]["port"], cmap="viridis", norm=midmap)

        # Bottom: Regeneration Functions
        ax_regen.clear()
        ax_regen.set_title("Regeneration functions")
        for idx, resource in enumerate(resources):
            t = torch.linspace(0, 2*np.pi, 1000)
            y = resource.regen_func(t+period-np.pi/2).numpy()
            ax_regen.plot(t+period, y, color=resource_colors[idx], label=f"Resource {idx+1}")

        ax_regen.axvline(x=period + np.pi/2, color='black', linestyle='--')  # Static vertical line
        ax_regen.axvspan(period, period + np.pi/2, facecolor='gray', alpha=0.2)  # Highlight indicating past
        ax_regen.legend(loc="upper right")

        ax_obstacle.clear()
        ax_obstacle.set_title("Obstacles")
        ax_obstacle.imshow(data[i]["obstacles"], cmap='gray')

        # show capital with colorbar
        ax_capital.clear()
        ax_capital.set_title("Capital")
        ax_capital.imshow(data[i]["capital"], cmap='viridis')

        ax_activation.clear()
        ax_activation.set_title("Activation")
        ax_activation.imshow(data[i]["actuators"][0], cmap='viridis')

        ax_waste.clear()
        ax_waste.set_title("Waste")
        ax_waste.imshow(data[i]["waste"], cmap='viridis') 
        
        # ax_resources.imshow(data[i]["resource_map"], cmap='tab20b', vmin=0, vmax=3)

    ani = FuncAnimation(fig, update_ani, frames=[i for i in range(len(periods))],interval=50, repeat=False)
    plt.tight_layout()
    plt.show()

animate()


def quiv_vis():
    # base_folder = "runs/"
    # stamp = "20231005-0516"
    # folder = os.path.join(base_folder, stamp)
    # data = []
    # for i in range(100):
    #     path = os.path.join(folder, f"{i}.pt")
    #     if os.path.exists(path):
    #         data.append(torch.load(path))
    """
    muscle_radii torch.Size([7, 10, 10])
    capital torch.Size([10, 10])
    waste torch.Size([10, 10])
    port torch.Size([10, 10])
    resource_map torch.Size([10, 10])
    obstacles torch.Size([10, 10])
    actuators torch.Size([13, 10, 10])
    """

    """
    state_tensor[0] = np.random.uniform(low=-1, high=1, size=(10, 10)) # activations

    state_tensor[1] += np.random.uniform(low=0, high=1, size=(10, 10)) # cytoplasm
    state_tensor[1] = state_tensor[1] / np.linalg.norm(state_tensor[1])

    state_tensor[2:7] += np.random.uniform(low=-1, high=1, size=(5, 10, 10)) # muscle radii
    state_tensor[2:7] = state_tensor[2:7] / np.linalg.norm(state_tensor[2:7])
    """

    tensors = []
    for d in data:
        state_tensor = np.empty((7, cfg.world_shape[0], cfg.world_shape[1]))
        state_tensor[0] = d["actuators"][actuator_indecies["flow_act"]]
        state_tensor[1] = d["capital"]
        state_tensor[2:7] = d["muscle_radii"][1:6]
        tensors.append(state_tensor)

    # Make the UV quiver thing a function so it can be any 3x3 kernel
    # Muscle value can be > 1
    # Why are some long arrows black? Should be colored?

    # Load the 100 tensors from the directory

    WIDTH = len(tensors[0][0][0])
    HEIGHT = len(tensors[0][0])

    plots = []

    fig, ax = plt.subplots()

    def update_plot(frame):
        def muscle_activations(muscle_radii, activations):
            return muscle_radii * activations
        
        def get_muscle_color(value):
            if value > 0:
                green = value * 255
                red = 0
            else:
                green = 0
                red = -value * 255
            return (red, green, 0)  # Return RGB tuple
        
        ax.clear()

        X, Y = np.meshgrid(range(0,WIDTH), range(0,HEIGHT))
        quiver_scale = WIDTH * .005
        
        # Quiver Plots (Muscle Radii * Activation)
        UP = muscle_activations(tensors[frame][3], tensors[frame][0])
        U = np.zeros_like(UP)
        V = np.abs(UP)
        ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in UP.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

        RIGHT = muscle_activations(tensors[frame][4], tensors[frame][0])
        U = np.abs(RIGHT)
        V = np.zeros_like(RIGHT)
        ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in RIGHT.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

        DOWN = muscle_activations(tensors[frame][5], tensors[frame][0])
        U = np.zeros_like(DOWN)
        V = -np.abs(DOWN)
        ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in DOWN.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

        LEFT = muscle_activations(tensors[frame][6], tensors[frame][0])
        U = -np.abs(LEFT)
        V = np.zeros_like(LEFT)
        ax.quiver(X, Y, U, V, color=[get_muscle_color(value) for value in LEFT.flatten()], angles='xy', scale_units='xy', scale=quiver_scale)

        # Scatter Plot (Cytoplasm)
        cytoplasm = tensors[frame][1]
        ax.scatter(X, Y, color=(.18, .97, 1), alpha=.3, s=[value*5000 for value in cytoplasm.flatten()])

        ax.set_xlim(-1, WIDTH)
        ax.set_ylim(-1, HEIGHT)    


        ani = animation.FuncAnimation(fig=fig, func=update_plot, frames=40, interval=100)
        f = r"./limited_eincasm_20.gif" 
        writergif = animation.PillowWriter(fps=20) 
        ani.save(f, writer=writergif)

quiv_vis()