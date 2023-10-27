import taichi as ti
from src_ti.taichi_world import World

class eincasm:
    def __init__(self, w=100, h=100, num_com=5, flow_kernel=None, ti_arch=ti.gpu):
        self.ti_arch = ti_arch
        self.torch_device = torch_device
        ti.init(arch=self.ti_arch)
        if flow_kernel is None:
            flow_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
            flow_kernel[0] = [0, 0]  # ORIGIN
            flow_kernel[1] = [-1, 0] # UP
            flow_kernel[2] = [0, 1]  # RIGHT
            flow_kernel[3] = [1, 0]  # DOWN
            flow_kernel[4] = [0, -1] # LEFT
        self.flow_kernel = flow_kernel
        self.w, self.h = w, h
        self.num_com = num_com

        self.world, self.sensors, self.actuators = self.define_world()
        self.world.malloc_torch()
        # TODO: A bit silly since Torch creates new tensors for conv2d - switch to taichi NN? LOL
        # Necessary to generate the index tree
        self.actuators.malloc_torch()

        self.paused = False
        self.brush_radius = 5
        self.drawing = False
        self.perturbing_weights = False
        self.perturbation_strength = 0.1
        self.noise_strength = 0.01
        self.visualize = True

        self.random_org = nn.Conv2d(
            self.world.data[self.sensors].shape[0],
            self.actuators.mem.shape[0],
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=self.torch_device,
        )


    def define_world(self):
        self.world = World(
            shape = (self.w, self.h),
            dtype = torch.float32,
            torch_device = self.torch_device,
            channels = {
                'muscles': ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32,),
                'capital':  {'lims': (0,10)},
                'waste':    {'lims': (0,1)},
                'obstacle': {'lims': (0,1), 'init': pcg.init_obstacles_perlin},
                'port': {
                    'lims': (-1,10),
                    'init': pcg.init_ports_levy,
                    'metadata': {
                        'num_resources': 2,
                        'min_regen_amp': 0.5,
                        'max_regen_amp': 2,
                        'alpha_range': [0.4, 0.9],
                        'beta_range': [0.8, 1.2],
                        'num_sites_range': [2, 10]},},
                'com': ti.types.vector(n=self.num_com, dtype=ti.f32)})
        
        self.sensors = ['capital', 'waste', 'obstacle', 'com']
        
        # TODO: This unnecessarily initialized the memory for the actuators - unneeded since the organism produced this and it will be overwrit
        self.actuators = World(
            shape = (self.w, self.h),
            dtype = torch.float32,
            torch_device = self.torch_device,
            channels = {
                'com': ti.types.vector(n=self.num_com, dtype=ti.f32),
                'macts': ti.types.struct(
                    flow=ti.f32,
                    port=ti.f32,
                    mine=ti.f32),
                'gracts': ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32)})
        
        # self.world.add_channel()
    
        return self.world, self.sensors, self.actuators

    def apply_rules(self):
        self.actuators.mem = self.random_org(self.world.data[self.sensors].unsqueeze(0))
        # self.actuators.mem.



ein = eincasm(w=3,h=3)
print(ein.world.data['obstacle'])