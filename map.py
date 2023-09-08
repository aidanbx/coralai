import generate_env as gen
import simulate_lifecycle as sim











def main():
    env_channels = gen.generate_env()
    config = sim.load_check_config("./config.yaml")
    live_channels = sim.init_live_channels(config)
    sim.inoculate_env(config, env_channels, live_channels)


if __name__ == "__main__":
    main()