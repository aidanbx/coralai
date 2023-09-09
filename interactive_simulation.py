import yaml

import generate_env as genv
import simulate_lifecycle as sim











def main():
    with open('./config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    env_channels = genv.generate_env()
    config = sim.load_check_config("./config.yaml")
    live_channels = sim.init_live_channels(config)
    sim.inoculate_env(config, env_channels, live_channels)


if __name__ == "__main__":
    main()