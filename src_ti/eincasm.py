import taichi as ti
import torch
from src_ti.world import World

@ti.data_oriented
class eincasm:
    def __init__(self, shape=None, torch_device=torch.device('cpu'),
                 num_com=5, flow_kernel=None):
        if shape is None:
            shape = (100,100)
        self.shape = shape
        self.torch_device = torch_device
        
        if flow_kernel is None:
            flow_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
            flow_kernel[0] = [0, 0]  # ORIGIN
            flow_kernel[1] = [-1, 0] # UP
            flow_kernel[2] = [0, 1]  # RIGHT
            flow_kernel[3] = [1, 0]  # DOWN
            flow_kernel[4] = [0, -1] # LEFT

        self.flow_kernel = flow_kernel

        self.num_com = num_com
        self.world = self.world_def()

    def world_def(self):
        return World(
            shape = self.shape,
            torch_dtype=torch.float32,
            torch_device=self.torch_device,
            channels = {
                'muscles': ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32,),
                'capital':  {'lims': (0,10)},
                'waste':    {'lims': (0,1)},
                'obstacle': {'lims': (0,1)},
                'port': {
                    'lims': (-1,10),
                    'metadata': {
                        'num_resources': 2,
                        'min_regen_amp': 0.5,
                        'max_regen_amp': 2,
                        'alpha_range': [0.4, 0.9],
                        'beta_range': [0.8, 1.2],
                        'num_sites_range': [2, 10]},},
                'portmap': ti.i8,
                'macts': ti.types.struct(
                    flow=ti.f32,
                    port=ti.f32,
                    mine=ti.f32),
                'gracts': ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32),
                'com': ti.types.vector(n=self.num_com, dtype=ti.f32)})