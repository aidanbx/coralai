import os
import torch
import neat

import taichi as ti
import configparser
from datetime import datetime


from coralai.substrate.substrate import Substrate
from coralai.instances.minimal.minimal_vis import MinimalVis
from coralai.instances.minimal.minimal_organism_cppn import MinimalOrganism

SHAPE = (400, 400)

def define_substrate(shape):
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    substrate = Substrate(
        shape=shape,
        torch_dtype=torch.float32,
        torch_device=torch_device,
        channels={
            "bw": ti.f32,
        },
    )
    substrate.malloc()
    return substrate

def define_organism(substrate):
    sensors = ['bw']
    sensor_inds = substrate.windex[sensors]
    n_sensors = len(sensor_inds)
    return MinimalOrganism(batch_size=SHAPE[0] * SHAPE[1],
                            n_sensors = n_sensors,
                            n_actuators = 1,
                            sensor_names = sensors,
                            actuator_names = ['bw'],
                            torch_device = substrate.torch_device)


def main():
    substrate = define_substrate(SHAPE)
    organism = define_organism(substrate)
    vis = MinimalVis(substrate, ["bw"])

    while vis.window.running:
        substrate.mem = organism.forward(substrate.mem)
        vis.update()
        if vis.mutate:
            organism.mutate(vis.weight_mutate_rate, vis.weight_mutate_power, vis.bias_mutate_rate)


if __name__ == "__main__":
    main()