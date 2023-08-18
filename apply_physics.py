# %%
"""
- Cytoplasm absorption of nutrients from food sources
	- Each time step, all food sources on the same square as cytoplasm are depleted and turned into cytoplasm
	- How much volume of cytoplasm can a cell of food size 1 convert into?
	- How much volume of cytoplasm can be converted per each time step, (fixed number or percent of potential created volume)
	- If the reservoir of the cell is filled/close to full, this process doesn't happen/caps out
- Exchange rate for
	- Cytoplasm->nutrient storage
	- Nutrient storage<->muscle
	- Nutrient storage->cytoplasm exchange/muscle contraction (to move x% of your reservoir, how much cytoplasm storage do you consume)
	- Nutrient storage->cytoplasm (perhaps no loss for the exchange)
- note: to contract a muscle, there must sufficient nutrient storage present, all will be consumed if contraction is larger than available. Cytoplasm is converted to nutrient storage (to fulfill the minimum automatically and more if the cell desires) before performing this operation.
- Cytoplasm falling on new cells should always create a minimal amount of nutrient storage (slime) - this and hidden channels are left behind as markers
	- If nutrient storage is below this amount
		- cytoplasm is automatically converted
		- muscle is automatically converted
		- the cell does not update
- Physical parameters can change via weather and local rules (esp relevant in a unindividuated large world evolution) - not implemented in current project
- Food source regeneration should be a parameter - what are the implications for memory? - not implemented in current project
"""
config = None
cell = None 
actuators = None
env_channels = None
live_channels = None
physiology = None

def apply_local_physics(config, cell, actuators, env_channels, live_channels):
    absorb_convert_food(config, cell, env_channels, live_channels)
    exchange_muscle(config, cell, actuators, live_channels)
    exchange_nutrient_storage(config, cell, actuators, live_channels)
    contract_muscle(config, cell, actuators, env_channels, live_channels)
    delegate_cytoplasm()

def check_config(config):
    pass

def absorb_convert_food(config, cell, env_channels, live_channels):
    pass

def exchange_muscle(config, cell, actuators, live_channels):
    pass

def exchange_nutrient_storage(config, cell, actuators, live_channels):
    pass 

def contract_muscle(config, cell, actuators, env_channels, live_channels):
    pass

def delegate_cytoplasm():
    pass


# %%