# %%
import yaml
import importlib
import tester
importlib.reload(tester)

import numpy as np

# %% Load Config & init Tests -------------------------------------------------
def load_check_config(config_object):
    if isinstance(config_object, str):
        with open(config_object) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    elif isinstance(config_object, dict):
        config = config_object
    else:
        raise TypeError("config_object must be either a path (str) or a config (dict)")

    all_visible_channels = (config["environment"]["channels"]
                    + config["physiology"]["channels"]
                    + config["physiology"]["actuators"])

    # Ensure all perception and action channels exist
    for channel in config["physiology"]["perception_channels"]:
        assert channel in all_visible_channels, f"Channel {channel} not found in all_channels"
        
    assert config["environment"]["channels"] == ["food", "poison", "obstacle"]
    assert config["physiology"]["channels"] == ["capital", "storage", "muscle"]
    assert config["physiology"]["perception_channels"] == ["food", "poison", "obstacle", "capital", "storage", "muscle"]
    assert config["physiology"]["actuators"] == ["rotate", "contract", "delta_muscle", "delta_storage"] 
    return config

if __name__ == "__main__":
    verbose = True # For testing
    config = tester.test(lambda: load_check_config("./config.yaml"),
        "Load Check Config Test", verbose=verbose)

    # cell: location
    # actuators: cell output
    # env_channels and live_channels: shape determined by locality kernel/ajacency matrix?
    cell = np.array([1, 1])
    actuator_names = config["physiology"]["actuators"]
    env_channel_names = config["environment"]["channels"]
    live_channel_names = config["physiology"]["channels"]

    actuators = np.array([+1.5,      # rotate 
                            -0.8,    # contraction
                            +0.1,    # delta_muscle
                            +0.1])   # delta_storage
    adjacency_kernel = config["physics"]["adjacency_kernel"]
    if adjacency_kernel == "moore":
        # env and live channels of moore kernel shape (3x3)
        env_channels = np.array([[0, 0, 0], [0.8, 0, 0], [0, 0, 0],   # food channel
                                [0, 0, 0], [0, 0, 0], [0, 0, 0],     # poison channel
                                [0, 1, 1], [0, 0, 0], [0, 0, 0]])      # obstacle channel, binary
        
        live_channels = np.array([[0.2, 0, 0], [0.8, 1, 0], [0, 0, 0],   # capital channel
                                [0, 0, 0], [0.1, 0, 1], [0, 0, 0],     # storage channel
                                [0, 1, 1], [0, 0, 0], [0, 0, 0]])      # muscle channel


if __name__ == "__main__":
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


"""
- capital absorption from food sources
	- Each time step, all food sources on the same square as capital are depleted and turned into capital
	- How much volume of capital can a cell of food size 1 convert into?
	- How much volume of capital can be converted per each time step, (fixed number or percent of potential created volume)
	- If the reservoir of the cell is filled/close to full, this process doesn't happen/caps out
- Exchange rate for
	- capital->storage
	- storage<->muscle
	- storage->capital exchange/muscle contraction (to move x% of your reservoir, how much capital storage do you consume)
	- storage->capital (perhaps no loss for the exchange)
- note: to contract a muscle, there must sufficient storage present, all will be consumed if contraction is larger than available. capital is converted to storage storage (to fulfill the minimum automatically and more if the cell desires) before performing this operation.
- capital falling on new cells should always create a minimal amount of storage (slime) - this and hidden channels are left behind as markers
	- If storage is below this amount
		- capital is automatically converted
		- muscle is automatically converted
		- the cell does not update
- Physical parameters can change via weather and local rules (esp relevant in a unindividuated large world evolution) - not implemented in current project
- Food source regeneration should be a parameter - what are the implications for memory? - not implemented in current project
"""

"""
physics:
  capital_absorption_rate: 0.1
  exchange_rates:
    food_to_capital: 5.0        # 1 food -> 5 capital 
    capital_to_storage: 0.5    # 1 capital -> 0.2 storage
    storage_to_muscle: 0.7       # 1 storage -> 0.7 muscle
    muscle_contraction_cost: 0.2  
  min_storage: 0.1
  muscle_max_size: 2.0
  storage_max_units: 10
  muscle_to_storage_atrophy_rate: 0.1
"""
# %%

# %% Ensure Min Storage, Absorb Convert Food ----------------------------------
def ensure_min_storage(config, cell, live_channels):
    storage_on_cell = live_channels[config["physiology"]["channels"].index("storage")][cell[0], cell[1]]
    capital_on_cell = live_channels[config["physiology"]["channels"].index("capital")][cell[0], cell[1]]
    capital_to_storage_rate = config["physiology"]["exchange_rates"]["capital_to_storage"]

    if storage_on_cell < config["physiology"]["min_storage"]:
        storage_deficit = config["physiology"]["min_storage"] - storage_on_cell
        req_capital = storage_deficit / capital_to_storage_rate
        if req_capital > capital_on_cell:
            live_channels[config["physiology"]["channels"].index("storage")][cell[0], cell[1]] += capital_on_cell * capital_to_storage_rate
            live_channels[config["physiology"]["channels"].index("capital")][cell[0], cell[1]] = 0
        else:
            live_channels[config["physiology"]["channels"].index("storage")][cell[0], cell[1]] += storage_deficit
            live_channels[config["physiology"]["channels"].index("capital")][cell[0], cell[1]] -= req_capital


def absorb_convert_food(config, cell, env_channels, live_channels):
    food_on_cell = env_channels[config["environment"]["channels"].index("food")][cell[0], cell[1]]
    max_intake = 1 - live_channels[config["physiology"]["channels"].index("capital")][cell[0], cell[1]]

    if food_on_cell > 0:
        intake = max(min(max_intake, config["physics"]["capital_absorption_rate"] * food_on_cell), food_on_cell)
        live_channels[config["physiology"]["channels"].index("capital")][cell[0], cell[1]] += intake
        env_channels[config["environment"]["channels"].index("food")][cell[0], cell[1]] -= intake
    
    ensure_min_storage(config, cell, live_channels)