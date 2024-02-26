import os
from coralai.evolution.neat_xor_demo import run

def main():
    # Determine the path to the configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'coralai/evolution/neat.config')

    # Create a dedicated folder for checkpoints
    checkpoint_dir = 'xor_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_prefix = 'neat_xor_checkpoint_'

    # Create a special directory for output files
    output_dir = 'neat_output'
    os.makedirs(output_dir, exist_ok=True)

    # Run the NEAT XOR demo with custom output directory and prefix
    run(config_path, output_dir=output_dir, checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix)


if __name__ == '__main__':
    main()