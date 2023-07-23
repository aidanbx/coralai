

# %% 
import numpy as np
import noise
from scipy.stats import uniform
from scipy.stats import levy_stable
from scipy import signal
import yaml

import importlib
import visualize
importlib.reload(visualize)

# %% 

def generate_env(config_file, visualize=False):
    config = load_check_config(config_file)
    env_channels = init_env_channels(config)
    populate_env_channels(config, env_channels)
    remove_overlap(config, env_channels)
    populate_chemoattractant(config, env_channels)
    populate_chemorepellant(config, env_channels)
    remove_overlap(config, env_channels)

    if visualize:
        visualize_env(config_file, env_channels)
    return env_channels

    

def load_check_config(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        # asserts alpha and beta are in their proper ranges
        assert 0 < config["environment"]["food_generation_params"]["alpha"] <= 2
        assert -1 <= config["environment"]["food_generation_params"]["beta"] <= 1
        assert 0 < config["environment"]["poison_generation_params"]["alpha"] <= 2
        assert -1 <= config["environment"]["poison_generation_params"]["beta"] <= 1

        return config
    

def init_env_channels(config):
    width = config["environment"]["width"]
    height = config["environment"]["height"]
    n_env_channels = len(config["environment"]["channels"])
    env_channels = np.zeros((n_env_channels, width, height))
    return env_channels


def populate_env_channels(config, env_channels):
    for ch in config["environment"]["channels"]:
        channel = env_channels[config["environment"]["channels"].index(ch)]
        if ch == "food":
            populate_food(config, channel)
        elif ch == "poison":
            populate_poison(config, channel)
        elif ch == "obstacle":
            populate_obstacle(config, channel)
        elif ch == "chemoattractant":
            pass
        elif ch == "chemorepellant":
            pass
        else:
            raise ValueError(f"Channel {ch} not recognized.")
    return env_channels


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
        threshold = obstacle_params.get("threshold", 0.1)  # default threshold is 0.5
        frequency = obstacle_params.get("frequency", 8)  # default frequency is 1.0
        
        for x in range(channel.shape[0]):
            for y in range(channel.shape[1]):
                # TODO add random offset, currently static
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


def populate_chemoattractant(config, env_channels):
    food_channel = env_channels[config["environment"]["channels"].index("food")]
    obstacle_channel = env_channels[config["environment"]["channels"].index("obstacle")]
    chemoattractant_channel = env_channels[config["environment"]["channels"].index("chemoattractant")]

    chemoattractant_channel += food_channel

    diffused_channel = diffuse_chemical(chemoattractant_channel, obstacle_channel, config["environment"]["chemoattractant_params"]["iterations"])
    env_channels[config["environment"]["channels"].index("chemoattractant")] = diffused_channel

def populate_chemorepellant(config, env_channels):
    poison_channel = env_channels[config["environment"]["channels"].index("poison")]
    obstacle_channel = env_channels[config["environment"]["channels"].index("obstacle")]
    chemorepellant_channel = env_channels[config["environment"]["channels"].index("chemorepellant")]

    chemorepellant_channel += poison_channel

    diffused_channel = diffuse_chemical(chemorepellant_channel, obstacle_channel, config["environment"]["chemorepellant_params"]["iterations"])
    env_channels[config["environment"]["channels"].index("chemorepellant")] = diffused_channel


def diffuse_chemical(channel, obstacle_channel, iterations):
       # iterations decide how much the chemical spreads out
    for _ in range(iterations):
        # Using convolution for averaging neighboring cells
        kernel = np.array([
            [0.3, 0.6, 0.3],
            [0.6, 1, 0.6],
            [0.3, 0.6, 0.3]
        ])
        new_channel = signal.convolve2d(channel, kernel, mode='same', boundary='wrap')
        
        # Ensure obstacles do not participate in diffusion
        new_channel[obstacle_channel > 0] = 0
        
        channel = new_channel
    return channel


def remove_overlap(config, env_channels):
    food_channel = env_channels[config["environment"]["channels"].index("food")]
    poison_channel = env_channels[config["environment"]["channels"].index("poison")]
    obstacle_channel = env_channels[config["environment"]["channels"].index("obstacle")]
    chemoattractant_channel = env_channels[config["environment"]["channels"].index("chemoattractant")]
    chemorepellant_channel = env_channels[config["environment"]["channels"].index("chemorepellant")]

    # 1. Clear other signals where there's an obstacle
    food_channel[obstacle_channel > 0] = 0
    poison_channel[obstacle_channel > 0] = 0
    chemoattractant_channel[obstacle_channel > 0] = 0
    chemorepellant_channel[obstacle_channel > 0] = 0

    # 2. Where both food and poison signals exist, clear poison
    overlap = (food_channel > 0) & (poison_channel > 0)
    poison_channel[overlap] = 0



def visualize_env(config_file, env_channels):    
    vis_config = visualize.load_check_config(config_file)

    food_index = vis_config["environment"]["channels"].index("food")
    poison_index = vis_config["environment"]["channels"].index("poison")
    obstacle_index = vis_config["environment"]["channels"].index("obstacle")

    images = visualize.channels_to_images(vis_config, env_channels)

    chemoattractant_image = images[vis_config["environment"]["channels"].index("chemoattractant")]
    chemorepellant_image = images[vis_config["environment"]["channels"].index("chemorepellant")]
    blended_image = 0.5 * (chemoattractant_image + chemorepellant_image)
    blended_image[...,3] = vis_config["visualize"]["chemo_alpha"]
    blended_image[(env_channels[obstacle_index] > 0), 3] = 0
    blended_image[(env_channels[poison_index] > 0), 3] = 0
    blended_image[(env_channels[food_index] > 0), 3] = 0

    images[food_index][env_channels[food_index] == 0, 3] = 0
    images[poison_index][env_channels[poison_index] == 0, 3] = 0
    images[obstacle_index][env_channels[obstacle_index] == 0, 3] = 0
    
    for image in images:
        visualize.show_image(image)
    visualize.show_image(blended_image)

    new_images = np.array([
        images[vis_config["environment"]["channels"].index("obstacle")],
        images[vis_config["environment"]["channels"].index("food")],
        images[vis_config["environment"]["channels"].index("poison")],
        blended_image,
    ])

    super_image = visualize.collate_channel_images(vis_config, new_images)
    visualize.show_image(super_image)


env_channels = generate_env("config.yaml", visualize=True)
# %%

