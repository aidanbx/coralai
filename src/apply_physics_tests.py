from apply_physics import *

import importlib
import simulate_lifecycle as life
importlib.reload(life)
import generate_env as genv
importlib.reload(genv)
# TEST: 0
# ---------------

env_channels = genv.generate_env("./config.yaml", visualize=True)
lifecycle_config = life.load_check_config("./config.yaml")
live_channels = life.init_live_channels(lifecycle_config)
cells_to_update = life.identify_cells_to_update(lifecycle_config, env_channels, live_channels)
life.inoculate_env(lifecycle_config, env_channels, live_channels)
cell = cells_to_update[0]
input = life.perceive(lifecycle_config, cell, env_channels, live_channels)

physiology = life.create_dumb_physiology(lifecycle_config)

output = life.act(lifecycle_config, input, physiology)

apply_local_physics(lifecycle_config, cell, output, env_channels, live_channels)   
# ---------------