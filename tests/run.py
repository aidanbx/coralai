import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, "../src/")
sys.path.append(module_dir)

import generate_env as genv
import simulate_lifecycle as sim
import eincasm_config

def main():
    config = eincasm_config.Config('./config.yaml')
    
    env_channels = genv.generate_env(config)

    physiology = sim.create_dumb_physiology(config)
    env_channels, live_channels = sim.simulate_lifecycle(config, env_channels, physiology)
    print("what we doin now")

if __name__ == "__main__":
    main()