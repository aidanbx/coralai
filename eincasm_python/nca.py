import taichi as ti
import torch

from .substrate.world import World
from .dynamics.organism_torch import Organism
from .ein_params import EinParams


@ti.data_oriented
class NCA:
    def __init__(
        self,
        params: EinParams = None,
        shape=None,
        torch_device=None,
    ):
        if params is None:
            params = EinParams()
        if shape is None:
            shape = (100, 100)
        if torch_device is None:
            torch_device = torch.device("cpu")

        self.params = params
        self.shape = shape
        self.torch_device = torch_device
        self.world = self.world_def()
        self.world.malloc()
        self.sensors = ['com']
        self.organism = Organism(self.world,
                                 sensors = self.sensors,
                                 n_actuators = self.world.windex['com'].shape[0])
        self.actions = None

    def world_def(self):
        return World(
            shape=self.shape,
            torch_dtype=torch.float32,
            torch_device=self.torch_device,
            channels={
                "com": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            },
       )
    