import taichi as ti
import torch

from ..substrate.world import World

@ti.data_oriented
class coralai:
    def __init__(
        self,
        shape=None,
        torch_device=None,
        num_com=None,
        flow_kernel=None,
    ):
        if shape is None:
            shape = (100, 100)
        if torch_device is None:
            torch_device = torch.device("cpu")
        if num_com is None:
            num_com = 8
        if flow_kernel is None:
            flow_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
            flow_kernel[0] = [0, 0]  # ORIGIN
            flow_kernel[1] = [-1, 0]  # UP
            flow_kernel[2] = [0, 1]  # RIGHT
            flow_kernel[3] = [1, 0]  # DOWN
            flow_kernel[4] = [0, -1]  # LEFT

        self.shape = shape
        self.w = shape[0]
        self.h = shape[1]
        self.torch_device = torch_device
        self.flow_kernel = flow_kernel
        self.num_com = num_com
        self.world = self.world_def()
        self.world.malloc()
        self.timestep = 0

    def apply_physics(self):
        # physics.activate_flow_muscles(self.world, self.flow_kernel, self.params.flow_cost)
        self.apply_ti_physics(self.world.mem, self.world.ti_indices)

    @ti.func
    def calculate_total_mass_and_gract(self, mem, inds, i, j):
        total_mass = 0
        total_gract = 0
        for k in range(inds.muscle.n):
            total_mass += mem[i, j, inds.com + k]
            total_gract += mem[i, j, inds.muscle_acts + k]
        return total_gract, total_mass

    @ti.kernel
    def apply_ti_physics(self, mem: ti.types.ndarray(), ti_inds: ti.template()):
        inds = ti_inds[None]
        for i, j in ti.ndrange(mem.shape[0], mem.shape[1]):
            capital = mem[i, j, inds.capital]
            total_gract, total_mass = self.calculate_total_mass_and_gract(mem, inds, i, j)
            mem[i, j, inds.total_mass] = total_mass


    # def init_channels(self):
    #     self.world["obstacle"] = pcg.init_obstacles_perlin(
    #         self.shape, self.world.channels["obstacle"].metadata
    #     )


    def world_def(self):
        return World(
            shape=self.shape,
            torch_dtype=torch.float32,
            torch_device=self.torch_device,
            channels={
                "energy": {"lims": [0, 1024]},
                "infra": {"lims": [0, 1024]},
                "com": {"lims": [-1,1]}
            },
       )
    