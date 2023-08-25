# %%
import yaml

def ensure_min_storage(config, cell, live_channels):
    storage_on_cell = live_channels[config["physiology"]["channels"].index("storage")][cell[0], cell[1]]
    cytoplasm_on_cell = live_channels[config["physiology"]["channels"].index("cytoplasm")][cell[0], cell[1]]
    cytoplasm_to_storage_rate = config["physiology"]["exchange_rates"]["cytoplasm_to_storage"]

    if storage_on_cell < config["physiology"]["min_storage"]:
        storage_deficit = config["physiology"]["min_storage"] - storage_on_cell
        req_cytoplasm = storage_deficit / cytoplasm_to_storage_rate
        if req_cytoplasm > cytoplasm_on_cell:
            live_channels[config["physiology"]["channels"].index("storage")][cell[0], cell[1]] += cytoplasm_on_cell * cytoplasm_to_storage_rate
            live_channels[config["physiology"]["channels"].index("cytoplasm")][cell[0], cell[1]] = 0
        else:
            live_channels[config["physiology"]["channels"].index("storage")][cell[0], cell[1]] += storage_deficit
            live_channels[config["physiology"]["channels"].index("cytoplasm")][cell[0], cell[1]] -= req_cytoplasm


def absorb_convert_food(config, cell, env_channels, live_channels):
    food_on_cell = env_channels[config["environment"]["channels"].index("food")][cell[0], cell[1]]
    max_intake = 1 - live_channels[config["physiology"]["channels"].index("cytoplasm")][cell[0], cell[1]]

    if food_on_cell > 0:
        intake = max(min(max_intake, config["physics"]["cytoplasm_absorption_rate"] * food_on_cell), food_on_cell)
        live_channels[config["physiology"]["channels"].index("cytoplasm")][cell[0], cell[1]] += intake
        env_channels[config["environment"]["channels"].index("food")][cell[0], cell[1]] -= intake
    
    ensure_min_storage(config, cell, live_channels)


def check_config(config):
    pass


def apply_local_physics(config, cell, actuators, env_channels, live_channels):
    check_config(config)
    absorb_convert_food(config, cell, env_channels, live_channels)
    exchange_muscle(config, cell, actuators, live_channels)
    exchange_storage(config, cell, actuators, live_channels)
    contract_muscle(config, cell, actuators, env_channels, live_channels)
    delegate_cytoplasm()


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
- Cytoplasm absorption from food sources
	- Each time step, all food sources on the same square as cytoplasm are depleted and turned into cytoplasm
	- How much volume of cytoplasm can a cell of food size 1 convert into?
	- How much volume of cytoplasm can be converted per each time step, (fixed number or percent of potential created volume)
	- If the reservoir of the cell is filled/close to full, this process doesn't happen/caps out
- Exchange rate for
	- Cytoplasm->storage
	- storage<->muscle
	- storage->cytoplasm exchange/muscle contraction (to move x% of your reservoir, how much cytoplasm storage do you consume)
	- storage->cytoplasm (perhaps no loss for the exchange)
- note: to contract a muscle, there must sufficient storage present, all will be consumed if contraction is larger than available. Cytoplasm is converted to storage storage (to fulfill the minimum automatically and more if the cell desires) before performing this operation.
- Cytoplasm falling on new cells should always create a minimal amount of storage (slime) - this and hidden channels are left behind as markers
	- If storage is below this amount
		- cytoplasm is automatically converted
		- muscle is automatically converted
		- the cell does not update
- Physical parameters can change via weather and local rules (esp relevant in a unindividuated large world evolution) - not implemented in current project
- Food source regeneration should be a parameter - what are the implications for memory? - not implemented in current project
"""

"""
physics:
  cytoplasm_absorption_rate: 0.1
  exchange_rates:
    food_to_cytoplasm: 5.0        # 1 food -> 5 cytoplasm 
    cytoplasm_to_storage: 0.5    # 1 cytoplasm -> 0.2 storage
    storage_to_muscle: 0.7       # 1 storage -> 0.7 muscle
    muscle_contraction_cost: 0.2  
  min_storage: 0.1
  muscle_max_size: 2.0
  storage_max_units: 10
  muscle_to_storage_atrophy_rate: 0.1
"""
# %%