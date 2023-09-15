import yaml

import generate_env as genv
import simulate_lifecycle as sim

def main():
    with open('./config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    env_channels = genv.generate_env(config)

    physiology = sim.create_dumb_physiology(config)
    env_channels, live_channels = sim.simulate_lifecycle(config, env_channels, physiology)
    print("what we doin now")

if __name__ == "__main__":
    main()