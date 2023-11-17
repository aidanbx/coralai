import numpy as np
import taichi as ti
import torch

from .sim.World import World
from .sim.Organism import Organism
from .sim import pcg
from .sim import physics
from .EinParams import EinParams


@ti.data_oriented
class Eincasm:
    def __init__(
        self,
        params: EinParams,
        shape=None,
        torch_device=torch.device("mps:0"),
        num_com=16,
        flow_kernel=None,
    ):
        self.params = params
        if shape is None:
            shape = (100, 100)
        self.shape = shape
        self.w = shape[0]
        self.h = shape[1]
        self.torch_device = torch_device

        if flow_kernel is None:
            flow_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
            flow_kernel[0] = [0, 0]  # ORIGIN
            flow_kernel[1] = [-1, 0]  # UP
            flow_kernel[2] = [0, 1]  # RIGHT
            flow_kernel[3] = [1, 0]  # DOWN
            flow_kernel[4] = [0, -1]  # LEFT

        self.flow_kernel = flow_kernel
        self.num_com = num_com
        self.world = self.world_def()
        self.world.malloc()
        self.init_channels()
        self.timestep = 0
        self.cid, self.mids = 0, np.array([1, 2])
        self.sensors = ['capital', 'obstacle',('com', 'r'), ('com', 'g'), ('com', 'b'), ('com', 'a')]
        self.actuators = [('com', 'r'), ('com', 'g'), ('com', 'b'), ('com', 'a')]
        self.organism = Organism(self.world,
                                 sensors = self.sensors,
                                 actuators = self.actuators)


    def apply_physics(self):
        physics.activate_flow_muscles(self.world, self.flow_kernel, self.params.flow_cost)
        self.apply_ti_physics(self.world.mem, self.world.ti_indices)


    @ti.kernel
    def apply_ti_physics(self, mem: ti.types.ndarray(), ti_inds: ti.template()):
        for i, j in ti.ndrange(mem.shape[0], mem.shape[1]):
            total_gract = 0.0
            capital = mem[i, j, ti_inds.capital]
            delta_cap_total = 0.0
            for mid in ti.static(range(ti_inds.muscles.n)):
                total_gract += mem[i, j, ti_inds.growth_acts[mid]]
            for mid in ti.static(range(ti_inds.muscles.n)):
                muscle = mem[i, j, ti_inds.muscles[mid]]
                delta = mem[i, j, ti_inds.growth_acts[mid]]
                # Distributed across muscle activations - more fuel available to the muscles that want to grow the most
                # This approximates continuous growth for N timestemps with a strength distribution across directions
                cap_for_muscle = delta / total_gract * capital
                grow_out = physics.grow_muscle_csa_ti(
                    cap_for_muscle,
                    muscle,
                    delta,
                    self.params.growth_efficiency,
                    self.params.capital_per_work_growth
                )
                new_rad, delta_cap = grow_out
                delta_cap_total += delta_cap
                mem[i, j, ti_inds.muscles[mid]] += new_rad
            capital += delta_cap_total

            port = mem[i, j, ti_inds.port]
            port_out = physics.activate_port_muscles_ti(
                capital,
                port,
                mem[i, j, ti_inds.obstacle],
                mem[i, j, ti_inds.muscles_port],
                mem[i, j, ti_inds.muscle_acts_port],
                self.params.capital_per_work_mine,
            )
            delta_cap, delta_port = port_out
            mem[i, j, ti_inds.port] += delta_port
            delta_cap_total += delta_cap
            capital += delta_cap_total

            obstacle = mem[i, j, ti_inds.obstacle]
            waste = mem[i, j, ti_inds.waste]
            mine_out = physics.activate_mine_muscles_ti(
                capital,
                obstacle,
                waste,
                mem[i, j, ti_inds.muscles_mine],
                mem[i, j, ti_inds.muscle_acts_mine],
                self.params.capital_per_work_mine,
            )
            delta_cap, delta_ob, delta_waste = mine_out
            delta_cap_total += delta_cap
            mem[i, j, ti_inds.capital] += delta_cap_total
            mem[i, j, ti_inds.obstacle] += delta_ob
            mem[i, j, ti_inds.waste] += delta_waste


    def init_channels(self):
        # pass
        p, pmap, r = pcg.init_ports_levy(
            self.shape, self.world.channels["port"].metadata
        )
        self.world["port"], self.world["portmap"], self.resources = p, pmap, r
        self.world["obstacle"] = pcg.init_obstacles_perlin(
            self.shape, self.world.channels["obstacle"].metadata
        )


    def world_def(self):
        return World(
            shape=self.shape,
            torch_dtype=torch.float32,
            torch_device=self.torch_device,
            channels={
                "capital": {"lims": [0, 10]},
                "waste": {"lims": [0, 1]},
                "obstacle": {"lims": [0, 1]},
                "port": {
                    "lims": [-1, 10],
                    "metadata": {
                        "num_resources": 2,
                        "min_regen_amp": 0.5,
                        "max_regen_amp": 2,
                        "alpha_range": [0.4, 0.9],
                        "beta_range": [0.8, 1.2],
                        "num_sites_range": [2, 10],
                    },
                },
                "portmap": ti.i8,
                "muscles": ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32,
                ),
                "muscle_acts": ti.types.struct(flow=ti.f32, port=ti.f32, mine=ti.f32),
                "growth_acts": ti.types.struct(
                    flow=ti.types.vector(n=self.flow_kernel.shape[0], dtype=ti.f32),
                    port=ti.f32,
                    mine=ti.f32,
                ),
                "com": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32, a=ti.f32),
                # rest=ti.types.vector(n=self.num_com-3, dtype=ti.f32))
            },
        )
