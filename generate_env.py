

# %% 
import numpy as np
import noise
from scipy.stats import uniform
from scipy.stats import levy_stable
import yaml

import importlib
import visualize
importlib.reload(visualize)

# %% 

def generate_env(config_file, visualize=False):
    config = load_check_config(config_file)
    channels = init_channels(config)
    channels = populate_channels(config, channels)
    if visualize:
        visualize_env(config_file, channels)
    return channels


def visualize_env(config_file, channels):
    vis_config = visualize.load_check_config(config_file)

    # visualize.channels_to_image_with_colorbar(channels)
    images = visualize.channels_to_images(vis_config, channels)
    for image in images:
        visualize.show_image(image)


def load_check_config(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        # asserts alpha and beta are in their proper ranges
        assert 0 < config["environment"]["food_generation_params"]["alpha"] <= 2
        assert -1 <= config["environment"]["food_generation_params"]["beta"] <= 1
        assert 0 < config["environment"]["poison_generation_params"]["alpha"] <= 2
        assert -1 <= config["environment"]["poison_generation_params"]["beta"] <= 1

        return config
    

def init_channels(config):
    width = config["environment"]["width"]
    height = config["environment"]["height"]
    n_channels = len(config["environment"]["channels"])
    channels = np.zeros((n_channels, width, height))
    return channels


def populate_channels(config, channels):
    for ch in config["environment"]["channels"]:
        if ch == "food":
            populate_food(config, channels[0])
        elif ch == "poison":
            populate_poison(config, channels[1])
        elif ch == "obstacle":
            populate_obstacle(config, channels[2])
        else:
            raise ValueError(f"Channel {ch} not recognized.")
    return channels


def populate_food(config, channel):
    food_params = config["environment"]["food_generation_params"]
    if config["environment"]["food_generation"] == "levy_dust":
        food_dust = levy_dust(
            (channel.shape[0], channel.shape[1]),
            food_params["num_food"],
            food_params["alpha"],
            food_params["beta"],
            pad = food_params["pad"]
        )
        discretize_levy_dust(food_dust, channel, pad = food_params["pad"])
    else:
        raise ValueError(f"Food generation method {config['environment']['food_generation']} not recognized.")


def populate_poison(config, channel):
    poison_params = config["environment"]["poison_generation_params"]
    if config["environment"]["poison_generation"] == "levy_dust":
        poison_dust = levy_dust(
            (channel.shape[0], channel.shape[1]),
            poison_params["num_poison"],
            poison_params["alpha"],
            poison_params["beta"],
            pad = poison_params["pad"]
        )
        discretize_levy_dust(poison_dust, channel, pad = poison_params["pad"])
    else:
        raise ValueError(f"Poison generation method {config['environment']['poison_generation']} not recognized.")


def populate_obstacle(config, channel):
    obstacle_params = config["environment"]["obstacle_generation_params"]
    
    if config["environment"]["obstacle_generation"] == "perlin_noise":
        threshold = obstacle_params.get("threshold", 0.5)  # default threshold is 0.5
        frequency = obstacle_params.get("frequency", 1.0)  # default frequency is 1.0
        
        for x in range(channel.shape[0]):
            for y in range(channel.shape[1]):
                value = noise.pnoise2(x / frequency, y / frequency, octaves=2)
                channel[x, y] = 1 if value > threshold else 0
    else:
        raise ValueError(f"Obstacle generation method {config['environment']['obstacle_generation']} not recognized.")


def levy_dust(shape: tuple, points: int, alpha: float, beta: float, pad: int = 0) -> np.array:
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi)

    step_length = abs(levy_stable.rvs(alpha, beta, size=points))

    x = np.cumsum(step_length * np.cos(angle)) % (shape[0] - pad)
    y = np.cumsum(step_length * np.sin(angle)) % (shape[1] - pad)

    return np.array([x, y])


def discretize_levy_dust(dust: np.array, channel: np.array, pad: int = 0) -> np.array:
    """Discretize a levy dust cloud into a grid of shape shape,
    such that each position in the grid is the number of points in the cloud that fall in that position

    Returns:
        np.array: Grid of shape shape representing the density of points in the dust cloud
    """
    dust = np.array(dust, dtype=np.int64)
    points, density = np.unique(dust, axis=1, return_counts=True)
    channel[points[0,:] + int(pad/2), points[1,:] + int(pad/2)] = density


generate_env("config.yaml", visualize=True)
# %%

