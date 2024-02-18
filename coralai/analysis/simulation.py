from ..dynamics.Organism import Organism
from ..dynamics.Physics import Physics
from ..substrate.world import World
from .visualization import Visualization

class Simulation:
    def __init__(self, 
                 world: World,
                 physics: Physics,
                 organism: Organism,
                 visualization: Visualization = None,):
        self.world = world
        self.physics = physics
        self.organism = organism
        self.vis = visualization

    def run(self):
        if self.vis is not None:
            while self.vis.window.running:
                if self.vis.perturbing_weights:
                    self.organism.perturb_weights(self.vis.perturbation_strength)
                # self.organism.forward(self.world.mem)
                self.physics.apply_actuators(self.world, self.organism.forward(self.world.mem))
                self.vis.update()