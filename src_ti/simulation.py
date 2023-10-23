import taichi as ti
import torch
import numpy as np
import src_ti.world as world

class Rule:
    def __init__(self, id, func, metadata: dict=None, **kwargs):
        self.id = id
        self.func = func
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.metadata['id'] = self.id

    def apply(self, sim):
        self.func(sim)

@ti.data_oriented
class Simulation:
    def __init__(
            self,
            id,
            world,
            metadata: dict = None, **kwargs):
        self.id = id
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.metadata['id'] = self.id
        self.world = world
        if self.world.memory_allocated:
            self.data = world.data
        else:
            self.data = None
    
    def start(self):
        self.world.malloc()
        self.data = self.world.data
