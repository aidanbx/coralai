import os
import torch
import taichi as ti

from coralai.coralai_cor import CoralaiCor
from coralai.population import Population


def produce_alternating_order(len, torch_device):
    order = []
    ind = 0
    i = 0
    while i < len:
        order.append(ind)
        if ind > 0:
            ind = -ind
        else:
            ind = (-ind + 1)
        i += 1
    return torch.tensor(order, device = torch_device)


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    kernel = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=torch_device) # ccw
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, "./experiment/coral_neat.config")
    cor = CoralaiCor(
        config_path = config_path,
        channels = {
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=4, dtype=ti.f32) # no, forward, left, right
            ),
            "com": ti.types.struct(
                a=ti.f32,
                b=ti.f32,
                c=ti.f32,
                d=ti.f32
            ),
            "rot": ti.f32,
            "genome": ti.f32,
            "genome_inv": ti.f32
        },
        shape = (100, 100),
        kernel = kernel[produce_alternating_order(kernel.shape[0], torch_device), :], # center, left, right, ll, rr, etc.
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )

    dir_order = [0, -1, 1, -2], # forward (with rot), left of rot, right of rot, behind
    pop = Population(cor, num_genomes=10)
    pass
    # visualizer = CoralVis(cor)
    # physics = PhysicsA(dir_order = dir_order)
    # population = Population()
    # culler = Culler(cor, population)
    # mutator = RadMutator(cor, population)

    # experiment = Experiment(cor, population, mutator, culler, physics, visualizer)
    # experiment.run()

