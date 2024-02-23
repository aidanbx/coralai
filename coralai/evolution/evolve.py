import os
import neat
import torch
import taichi as ti
from coralai.instances.coral.coral_organism import CoralOrganism
from coralai.substrate.substrate import Substrate
from coralai.simulation.simulation import Simulation
from coralai.instances.coral.coral_physics import CoralPhysics
from coralai.simulation.evolver import Evolver

class CoralEvolver(Evolver):
    def __init__(self, config_path):
        super(CoralEvolver, self).__init__()

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

        # Initialize Taichi and Torch
        ti.init(ti.metal)
        torch_device = torch.device("mps")

        # Define the world and physics (shared across all organisms)
        shape = (400, 400)
        N_HIDDEN_CHANNELS = 8
        world = Substrate(
            shape=shape,
            torch_dtype=torch.float32,
            torch_device=torch_device,
            channels={
                "energy": ti.f32,
                "infra": ti.f32,
                "last_move": ti.f32,
                "com": ti.types.vector(n=N_HIDDEN_CHANNELS, dtype=ti.f32),
            },
        )
        world.malloc()
        physics = CoralPhysics()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Convert NEAT genome to CoralOrganism
        organism = CoralOrganism(world,
                                 sensors=['energy', 'infra', 'last_move', 'com'],
                                 n_actuators=1 + 1 + 1 + N_HIDDEN_CHANNELS)
        # Here you would typically convert genome to organism parameters

        # Run the simulation without visualization for N timesteps
        sim = Simulation(world, physics, organism, None)  # No visualization
        for _ in range(N_TIMESTEPS):
            sim.step()

        # Calculate fitness as the sum of infrastructure
        fitness = torch.sum(world.mem[:, world.channels['infra'], :, :]).item()
        genome.fitness = fitness

# Create the population
pop = neat.Population(config)

# Add a reporter to show progress in the terminal
pop.add_reporter(neat.StdOutReporter(True))
pop.add_reporter(neat.StatisticsReporter())

# Run the NEAT algorithm to evolve genomes
winner = pop.run(eval_genomes, 300)  # Run for 300 generations or until a fitness threshold is reached