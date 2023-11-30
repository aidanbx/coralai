import taichi as ti
import torch

from eincasm.substrate.world import World
from eincasm.instances.rgb_vis import RGBVis
from eincasm.instances.nca_organism import NCAOrganism
from eincasm.analysis.simulation import Simulation

ti.init(ti.metal)
torch_device = torch.device("mps")

N_HIDDEN_CHANNELS = 8
shape = (400, 400)

world = World(shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=N_HIDDEN_CHANNELS, dtype=ti.f32),
        },)

world.malloc()

organism = NCAOrganism(world,
        sensors = ['rgb', 'hidden'],
        n_actuators = world.windex[['rgb', 'hidden']].shape[0])

vis = RGBVis(world, ['rgb'])

sim = Simulation(world, organism, vis)

sim.run()
