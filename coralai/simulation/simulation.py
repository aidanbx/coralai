from ..dynamics.organism import Organism
from ..dynamics.environment import Environment
from .visualization import Visualization

class Simulation:
    def __init__(self, 
                 environment: Environment,
                 organism: Organism,
                 visualization: Visualization = None,):
        self.organism = organism
        self.environment = environment
        self.vis = visualization

    def step(self):
        self.environment.step(self.organism)
        if self.vis is not None:
            self.vis.update()
            