import random

import numpy as np
import noise

from scipy import signal
from scipy.stats import levy_stable
from scipy.stats import uniform


if __name__ == "__main__":
    verbose = True # For testing


def remove_overlap(config: dict, env_channels: np.array) -> None: 
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


def diffuse_chemical(channel: np.array, obstacle_channel: np.array, iterations: int, dropoff: float = 0.5) -> np.array:
    
    # iterations decide how much the chemical spreads out
    for _ in range(iterations):
        # Using convolution for averaging neighboring cells
        kernel = np.array([dropoff/1.])
        kernel = np.array([
            [dropoff/1.4, dropoff, dropoff/1.4], # CHANGE KERNEL ------------------------------------------------------------
            [dropoff,     1,       dropoff],
            [dropoff/1.4, dropoff, dropoff/1.4]
        ])
        new_channel = signal.convolve2d(channel, kernel, mode='same', boundary='wrap') # CHANGE FUNCTION? ------------------------------------------------------------
        
        # Ensure obstacles do not participate in diffusion
        new_channel[obstacle_channel > 0] = 0
        
        channel = new_channel
    return channel


def populate_chemoattractant(config: dict, env_channels: np.array) -> None:
    food_channel = env_channels[config["environment"]["channels"].index("food")]
    obstacle_channel = env_channels[config["environment"]["channels"].index("obstacle")]
    chemoattractant_channel = env_channels[config["environment"]["channels"].index("chemoattractant")]

    chemoattractant_channel += food_channel

    diffused_channel = diffuse_chemical(chemoattractant_channel, obstacle_channel,
                                        config["environment"]["chemoattractant_params"]["iterations"],
                                        config["environment"]["chemoattractant_params"]["dropoff"])
    env_channels[config["environment"]["channels"].index("chemoattractant")] = diffused_channel


def populate_chemorepellant(config: dict, env_channels: np.array) -> None:
    poison_channel = env_channels[config["environment"]["channels"].index("poison")]
    obstacle_channel = env_channels[config["environment"]["channels"].index("obstacle")]
    chemorepellant_channel = env_channels[config["environment"]["channels"].index("chemorepellant")]

    chemorepellant_channel += poison_channel

    diffused_channel = diffuse_chemical(chemorepellant_channel, obstacle_channel,
                                        config["environment"]["chemorepellant_params"]["iterations"],
                                        config["environment"]["chemorepellant_params"]["dropoff"])
    env_channels[config["environment"]["channels"].index("chemorepellant")] = diffused_channel


def discretize_levy_dust(dust: np.array, channel: np.array, pad: int = 0) -> np.array:
    """Discretize a levy dust cloud into a grid of shape shape,
    such that each position in the grid is the number of points in the cloud that fall in that position

    Returns:
        np.array: Grid of shape shape representing the density of points in the dust cloud
    """
    dust = np.array(dust, dtype=np.int64)
    points, density = np.unique(dust, axis=1, return_counts=True)
    channel[points[0,:] + int(pad/2), points[1,:] + int(pad/2)] = density


def levy_dust(shape: tuple, points: int, alpha: float, beta: float, pad: int = 0) -> np.array:
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi)

    step_length = abs(levy_stable.rvs(alpha, beta, size=points))

    x = np.cumsum(step_length * np.cos(angle)) % (shape[0] - pad)
    y = np.cumsum(step_length * np.sin(angle)) % (shape[1] - pad)

    return np.array([x, y]) 


def populate_obstacle(config: dict, channel: np.array) -> None:
    obstacle_params = config["environment"]["obstacle_generation_params"]
    
    if config["environment"]["obstacle_generation"] == "perlin_noise":
        threshold_range = obstacle_params.get("threshold", [0.05, 0.2])
        frequency_range = obstacle_params.get("frequency", [4.0, 16.0])
        octaves_range = obstacle_params.get("octaves", [1, 4])
        persistence_range = obstacle_params.get("persistence", [0.25, 1.0])
        lacunarity_range = obstacle_params.get("lacunarity", [1.5, 3.0])

        # Generate random values based on provided ranges
        threshold = random.uniform(*threshold_range)
        
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
                channel[x, y] = 1 if value > threshold else 0
    else:
        raise ValueError(f"Obstacle generation method {config['environment']['obstacle_generation']} not recognized.")
    

def populate_poison(config: dict, channel: np.array) -> None:
    poison_params = config["environment"]["poison_generation_params"]
    
    if config["environment"]["poison_generation"] == "levy_dust":
        pad_range = poison_params.get("pad", [0, 5])
        alpha_range = poison_params.get("alpha", [0.5, 0.9])
        beta_range = poison_params.get("beta", [0.3, 0.7])
        num_poison_range = poison_params.get("num_poison", [80, 120])
        
        pad = random.randint(*pad_range)
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        num_poison = random.randint(*num_poison_range)

        poison_dust = levy_dust((channel.shape[0], channel.shape[1]), num_poison, alpha, beta, pad=pad)
        discretize_levy_dust(poison_dust, channel, pad=pad)
    else:
        raise ValueError(f"Poison generation method {config['environment']['poison_generation']} not recognized.")


def populate_food(config: dict, channel: np.array) -> None:
    food_params = config["environment"]["food_generation_params"]
    
    if config["environment"]["food_generation"] == "levy_dust":
        pad_range = food_params.get("pad", [0, 5])
        alpha_range = food_params.get("alpha", [0.8, 1.2])
        beta_range = food_params.get("beta", [0.8, 1.2])
        num_food_range = food_params.get("num_food", [250, 350])
        
        pad = random.randint(*pad_range)
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        num_food = random.randint(*num_food_range)

        food_dust = levy_dust((channel.shape[0], channel.shape[1]), num_food, alpha, beta, pad=pad)
        discretize_levy_dust(food_dust, channel, pad=pad)
    else:
        raise ValueError(f"Food generation method {config['environment']['food_generation']} not recognized.")


def populate_env_channels(config: dict, env_channels: np.array) -> np.array: 
    for ch in config["environment"]["channels"]:
            channel = env_channels[config["environment"]["channels"].index(ch)]
            if ch == "food":
                populate_food(config, channel)
            elif ch == "poison":
                populate_poison(config, channel)
            elif ch == "obstacle":
                populate_obstacle(config, channel)
            elif ch == "chemoattractant":
                populate_chemoattractant(config, env_channels)
            elif ch == "chemorepellant":
                populate_chemorepellant(config, env_channels)
            else:
                raise ValueError(f"Channel {ch} not recognized.")

    return env_channels


def init_env_channels(config: dict) -> np.array:
    width = config["environment"]["width"]
    height = config["environment"]["height"]
    env_channels_length = len(config["environment"]["channels"])
    env_channels = np.zeros((env_channels_length, width, height))
    return env_channels


def check_config(config_object: dict) -> None: 
    assert isinstance(config_object, dict), "config must be a dict object"

    # Assert if alpha and beta values are within the valid range

    # Food
    assert 0 < config_object["environment"]["food_generation_params"]["alpha"][0] <= 2
    assert config_object["environment"]["food_generation_params"]["alpha"][1] <= 2
    assert -1 <= config_object["environment"]["food_generation_params"]["beta"][0] <= 1
    assert -1 <= config_object["environment"]["food_generation_params"]["beta"][1] <= 1

    # Poison
    assert 0 < config_object["environment"]["poison_generation_params"]["alpha"][0] <= 2
    assert config_object["environment"]["poison_generation_params"]["alpha"][1] <= 2
    assert -1 <= config_object["environment"]["poison_generation_params"]["beta"][0] <= 1
    assert -1 <= config_object["environment"]["poison_generation_params"]["beta"][1] <= 1


def generate_env(config: dict, visualize: bool = False) -> np.array:
    check_config(config)

    env_channels = init_env_channels(config)
    populate_env_channels(config, env_channels)
    remove_overlap(config, env_channels)
 
    return env_channels