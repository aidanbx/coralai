import torch
import taichi as ti
from coralai.substrate.substrate import Substrate
from coralai.instances.coral.coral_physics import apply_actuators
from coralai.instances.coral.coral_vis import CoralVis
from coralai.instances.coral.coral_organism import CoralOrganism

SHAPE = (400, 400)
N_HIDDEN_CHANNELS = 8


def define_substrate(shape, n_hidden_channels):
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "energy": ti.f32,
            "infra": ti.f32,
            "last_move": ti.f32,
            "com": ti.types.vector(n=n_hidden_channels, dtype=ti.f32),
        },
    )
    substrate.malloc()
    return substrate

substrate = define_substrate(SHAPE, N_HIDDEN_CHANNELS)
sensors = ['energy', 'infra', 'last_move', 'com']
sensor_inds = substrate.windex[sensors]
n_sensors = len(sensor_inds)

organism = CoralOrganism(substrate = substrate,
                         sensors = ['energy', 'infra', 'last_move', 'com'],
                         n_actuators = 1 + 1 + 1 + N_HIDDEN_CHANNELS, # invest, liquidate, explore, hidden
                         torch_device = substrate.torch_device)
# organism = DumbOrg(world)

vis = CoralVis(substrate, ['energy', 'infra', 'last_move'])

while vis.window.running:
    apply_actuators(substrate, organism.forward(substrate.mem))
    vis.update()
    if vis.mutating:
        organism.perturb_weights(vis.perturbation_strength)
