from .coralai_cor import CoralaiCor
from .reportable import Reportable
from .population import Population

class Mutator(Reportable):
    def __init__(self, cor: CoralaiCor, population: Population, metadata):
        super().__init__(metadata)
        self.cor = cor
        self.population = population

    def apply_mutation(self):
        pass
