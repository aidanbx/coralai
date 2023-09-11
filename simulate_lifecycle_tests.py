from simulate_lifecycle import *

import yaml
import generate_env

# TEST: 0
# ---------------
with open("./config.yaml", "r") as yaml_file:
    config = yaml.safe_load(yaml_file)
env_channels = generate_env.generate_env(config, visualize=True)
check_config(config)
live_channels = init_live_channels(config)
inoculate_env(config, env_channels, live_channels)
run_lifecycle(config, env_channels, live_channels, create_dumb_physiology(config))
# ---------------


# KERNEL PLAYGROUND
world_size = 20
world = np.zeros((world_size, world_size))
world[world_size//2, world_size//2] = 1

# Cellular automata that spreads outwards
kernel = np.array([[1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])

# convolve world with kernel
world = signal.convolve2d(world, kernel, mode='same')

# visualize.show_image(visualize.channel_to_image(world, cmap="viridis")).show()