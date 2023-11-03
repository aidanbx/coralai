import numpy as np
import taichi as ti
import torch
from src_ti.world import World
from src_ti import physics, pcg

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
        self.world.malloc()
        self.init_channels()
        self.timestep = 0 
        self.cid, self.mids = 0, np.array([1,2])

    @ti.kernel
    def apply_ti_physics(self, mem: ti.types.ndarray()):
        physics.tst(self.cid, self.mids)
        # physics.grow_muscle_csa_ti(mem,
        #                            self.world.indices['capital'][0],
        #                            self.world.indices['muscles'], self.world.indices['gracts'],
        #                            1.0, 0.85, 0.1)
        # for i, j in ti.ndrange(mem.shape[0], mem.shape[1]):
        #     mem[i, j, self.world.indices['capital']] += 0.1

    def init_channels(self):
        p, pmap, r = pcg.init_ports_levy(self.shape, self.world.channels['port'].metadata)
        self.world['port'], self.world['portmap'], self.resources = p, pmap, r
        self.world['obstacle'] = pcg.init_obstacles_perlin(self.shape, self.world.channels['obstacle'].metadata)

    def world_def(self):
        return World(
            shape = self.shape,
            torch_dtype=torch.float32,
            torch_device=self.torch_device,
            channels = {
                'capital':  {'lims': [0,10]},
                'waste':    {'lims': [0,1]},
                'obstacle': {'lims': [0,1]},
                'port': {
                    'lims': [-1,10],
                    'metadata': {
                        'num_resources': 2,
                        'min_regen_amp': 0.5,
                        'max_regen_amp': 2,
                        'alpha_range': [0.4, 0.9],
                        'beta_range': [0.8, 1.2],
                        'num_sites_range': [2, 10]},},
                'portmap': ti.i8,
                'muscles': ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32,),
                'macts': ti.types.struct(
                    flow=ti.f32,
                    port=ti.f32,
                    mine=ti.f32),
                'gracts': ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32),
                'com': ti.types.vector(n=self.num_com, dtype=ti.f32)})