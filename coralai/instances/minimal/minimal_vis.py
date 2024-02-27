import taichi as ti
from ...substrate.substrate import Substrate
from ...simulation.visualization import Visualization


@ti.data_oriented
class MinimalVis(Visualization):
    def __init__(
        self,
        substrate: Substrate,
        chids: list,
        name: str = None,
        scale: int = None,
    ):
        super(MinimalVis, self).__init__(substrate, chids, name, scale)

        if self.chindices.n != 1:
            raise ValueError("Vis: ch_cmaps must have 1 channel for black and white visualization")