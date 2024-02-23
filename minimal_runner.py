import torch
import taichi as ti
from coralai.substrate.substrate import Substrate
from coralai.instances.minimal.minimal_vis import MinimalVis
from coralai.instances.minimal.minimal_organism import MinimalOrganism

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
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=n_hidden_channels, dtype=ti.f32),
        },
    )
    substrate.malloc()
    return substrate

def define_organism(substrate):
    sensors = ['rgb', 'hidden']
    sensor_inds = substrate.windex[sensors]
    n_sensors = len(sensor_inds)
    return MinimalOrganism(n_sensors = n_sensors,
                            n_actuators = 3 + N_HIDDEN_CHANNELS, # invest, liquidate, explore, hidden
                            torch_device = substrate.torch_device)

def main():
    substrate = define_substrate(SHAPE, N_HIDDEN_CHANNELS)
    organism = define_organism(substrate)
    vis = MinimalVis(substrate, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])

    while vis.window.running:
        substrate.mem = organism.forward(substrate.mem)
        vis.update()
        if vis.perturbing_weights:
            organism.perturb_weights(vis.perturbation_strength)


if __name__ == "__main__":
    main()