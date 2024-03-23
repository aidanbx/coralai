from .reportable import Reportable

class Experiment(Reportable):
    def __init__(self, cor: CoralaiCor, population: Population, mutator: Mutator,
                 culler: Culler, physics: Physics, visualizer: Visualizer, metadata):
        super().__init__(metadata)
    
    

