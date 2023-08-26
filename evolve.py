import neat
import simulate_lifecycle

# Load NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neat.config')

# Create the population
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Add a checkpointer
p.add_reporter(neat.Checkpointer(5))

# Define the fitness function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simulate_lifecycle.simulate_lifecycle('config.yaml', env_channels, net)

# Run the NEAT algorithm
winner = p.run(eval_genomes, 300)