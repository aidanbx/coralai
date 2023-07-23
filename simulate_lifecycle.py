
# %%
import numpy as np
import yaml

import importlib
import generate_env
importlib.reload(generate_env)

def simulate_lifecycle(config_file, env_channels, physiology):
    config = load_check_config(config_file)
    life_channels = init_life_channels(config)
    innoculate_env(config, env_channels, life_channels)
    run_lifecycle(config, env_channels, life_channels, physiology)
    
    return env_channels, life_channels
    
def dumb_physiology(config):
    # Receives perceived channels and returns a random value, or maybe a hardcoded algorithm
    return 0

def innoculate_env(config, env_channels, life_channels):
    # Pick random food source, or multiple depending on config and add cytoplasm on it
    # or pick random points and add cytoplasm with nutrient source (seed/spore)
    return 0

def run_lifecycle(confgi, env_channels, life_channels, physiology):
    if config["lifecycle"]["reproduction_metric"] == "iterations":
        for _ in range(config["lifecycle"]["reproduction_params"]["iterations"]):
            # 
            d
    else:
        raise ValueError(f"Reproduction point {config['lifecycle']['reproduction_metric']} not recognized.")




channels = generate_env.generate_env("config.yaml", visualize=True)
simulate_lifecycle("config.yaml", channels, )
# %%