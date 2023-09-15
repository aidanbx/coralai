from simulate_lifecycle import *

import yaml
import numpy as np
from scipy import signal
import generate_env

# TEST: 0
# ---------------
def test0():
    with open("simulation/config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    env_channels = generate_env.generate_env(config, visualize=True)
    check_config(config)
    live_channels = init_live_channels(config)
    inoculate_env(config, env_channels, live_channels)
    run_lifecycle(config, env_channels, live_channels, create_dumb_physiology(config))
# ---------------

test0()