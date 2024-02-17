import taichi as ti
import torch

from coralai.substrate.world import World
from coralai.analysis.simulation import Simulation
from coralai.instances.coral_vis import CoralVis
from coralai.instances.coral_organism import CoralOrganism


N_HIDDEN_CHANNELS=8

ti.init(ti.metal)
torch_device = torch.device("mps")
shape = (400, 400)


world = World(
    shape=shape,
    torch_dtype=torch.float32,
    
    torch_device=torch_device,
    channels={
        "energy": ti.f32,
        "infra": ti.f32,
        "last_move": ti.f32, # 0, 0.5, 1
        "com": ti.types.vector(n=N_HIDDEN_CHANNELS, dtype=ti.f32),
    },
)

world.malloc()

organism = CoralOrganism(world,
    sensors = ['energy', 'infra', 'last_move', 'com'],
    n_actuators = 1 + 1 + 1 + N_HIDDEN_CHANNELS) # invest, liquidate, explore, hidden
print(world.windex)
# vis = CoralVis(world, ['rgb'])
vis = CoralVis(world, ['energy', 'infra', 'last_move'])

sim = Simulation(world, organism, vis)

sim.run()
