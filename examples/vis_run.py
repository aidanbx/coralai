from matplotlib import animation 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import importlib
import src.EINCASMConfig as EINCASMConfig
importlib.reload(EINCASMConfig)
cfg = EINCASMConfig.Config('config.yaml')

base_folder = "runs/"
stamp = "20231005-0516"
folder = os.path.join(base_folder, stamp)
data = []
for i in range(100):
    path = os.path.join(folder, f"{i}.pt")
    if os.path.exists(path):
       data.append(torch.load(path))
"""
muscle_radii torch.Size([7, 10, 10])
capital torch.Size([10, 10])
waste torch.Size([10, 10])
port torch.Size([10, 10])
resource_map torch.Size([10, 10])
obstacles torch.Size([10, 10])
actuators torch.Size([13, 10, 10])
"""
actuator_indecies = {
    "flow_act":  0,
    "mine_act":  1,
    "port_act":  2,
    "muscle_radii_delta":   [i+3                    for i in range(cfg.num_muscles)],
    "communication":        [i+3+cfg.num_muscles    for i in range(cfg.num_communication_channels)]
}

"""
state_tensor[0] = np.random.uniform(low=-1, high=1, size=(10, 10)) # activations

state_tensor[1] += np.random.uniform(low=0, high=1, size=(10, 10)) # cytoplasm
state_tensor[1] = state_tensor[1] / np.linalg.norm(state_tensor[1])

state_tensor[2:7] += np.random.uniform(low=-1, high=1, size=(5, 10, 10)) # muscle radii
state_tensor[2:7] = state_tensor[2:7] / np.linalg.norm(state_tensor[2:7])
"""

tensors = []
for d in data:
    state_tensor = np.empty((7, 10, 10))
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


