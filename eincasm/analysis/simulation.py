from ..dynamics.Organism import Organism
from ..substrate.world import World
from .visualization import Visualization


class Simulation:
    def __init__(self, 
                 world: World,
                 organism: Organism,
                 visualization: Visualization = None,):
        self.world = world
        self.organism = organism
        self.vis = visualization

    def run(self):
        if self.vis is not None:
            while self.vis.window.running:
                if self.vis.perturbing_weights:
                    self.organism.perturb_weights(self.vis.perturbation_strength)
                self.world.mem = self.organism.forward(self.world.mem)
                self.vis.update()