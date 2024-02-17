import torch
import taichi as ti
import torch.nn as nn

from ..dynamics.nn_lib import ch_norm
from ..dynamics.Organism import Organism


LIQUIDATE_IDX = 0
INVEST_IDX = 1
EXPLORE_IDX = 2
ACT_INDS = [LIQUIDATE_IDX, INVEST_IDX, EXPLORE_IDX]

@ti.data_oriented
class CoralOrganism(Organism):
    def __init__(self, world, sensors, n_actuators, latent_size = None):
        super(CoralOrganism, self).__init__(world, sensors, n_actuators)

        if latent_size is None:
            latent_size = (self.n_sensors + self.n_actuators) // 2
        self.latent_size = latent_size

        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_sensors,
            self.latent_size,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=self.world.torch_device,
            bias=False
        )

        # self.latent_conv = nn.Conv2d(
        #     self.latent_size,
        #     self.latent_size,
        #     kernel_size=3,
        #     padding=1,
        #     padding_mode='circular',
        #     device=self.world.torch_device,
        #     bias=False
        # )

        self.latent_conv_2 = nn.Conv2d(
            self.latent_size,
            self.n_actuators,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=self.world.torch_device,
            bias=False
        )

    @ti.kernel
    def distribute_energy(self, mem: ti.types.ndarray(), ti_inds: ti.template()):
        """
        For every given cell, transfer all of its energy to its neighbor with the highest infrastructure value
        
        Moore's neighborhood, including central cell
        Torus boundary condition
        """
        inds = ti_inds[None]
        for i, j in ti.ndrange(self.world.w, self.world.h):
            max_infra = mem[0, i, j, inds.infra]
            max_i, max_j = i, j
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                ni, nj = (i + di) % self.world.w, (j + dj) % self.world.h
                if mem[0, ni, nj, inds.infra] > max_infra:
                    max_infra = mem[0, ni, nj, inds.infra]
                    max_i, max_j = ni, nj
            if max_i != i or max_j != j:
                mem[0, max_i, max_j, inds.energy] += mem[0, i, j, inds.energy]
                mem[0, i, j, inds.energy] = 0.0


    # @ti.kernel
    # def apply_activations_to_world(self, mem: ti.types.ndarray(), x: ti.types.ndarray(), ti_inds: ti.template()):
    #     """
    #     Liquidate takes the infrastructure of the current cell and, in proportion to the activation of that cell, turns it into free energy at an exchange rate of 1:1. So an activation of 1.0 (thus meaning the other actuators have a value of 0.0 due to the softmax, which is unlikely) 100% of the infrastructure is converted into free energy.

    #     Invest takes free energy on the current cell and, just like liquidate, in proportion to the activation of that cell, turns it into infrastructure at an exchange rate of 1:1.

    #     Explore takes the free energy on the current cell and distributed a proportion of it as determined by the cell's activation to its neighbors, as determined by a moore's neighborhood
    #     """
    #     inds = ti_inds[None]
    #     for i, j in ti.ndrange(self.world.w, self.world.h):
    #         energy = mem[0, i, j, inds.energy]
    #         infra = mem[0, i, j, inds.infra]

    #         liquidate = x[0, 0, i, j]
    #         invest = x[0, 1, i, j]
    #         # explore = x[0, 2, i, j]

    #         # Apply liquidate operation
    #         energy += liquidate * infra
    #         infra -= liquidate * infra

    #         # Apply invest operation
    #         infra += invest * energy
    #         energy -= invest * energy

    #         # # Apply explore operation
    #         # for di, dj in ti.ndrange((-1, 2), (-1, 2)):  # Moore's neighborhood
    #         #     if di == 0 and dj == 0:
    #         #         continue  # Skip the current cell
    #         #     ni, nj = i + di, j + dj
    #         #     if 0 <= ni < self.world.w and 0 <= nj < self.world.h:
    #         #         mem[0, ni, nj, inds.infra] += explore * energy / 8  # Distribute equally to neighbors

    #         mem[0, i, j, inds.energy] = energy
    #         mem[0, i, j, inds.infra] = infra


    def forward(self, x):
        inds = self.world.ti_indices[None]
        with torch.no_grad():
            x = self.conv(x)
            x = nn.ReLU()(x)
            # x = ch_norm(x)
            # x = torch.sigmoid(x)

            # x = self.latent_conv(x)
            # x = nn.ReLU()(x)
            # x = ch_norm(x)
            # x = torch.sigmoid(x)

            x = self.latent_conv_2(x)

            x[:, ACT_INDS, :, :] = torch.softmax(x[:, ACT_INDS, :, :], dim=1)

            max_actuator = torch.argmax(x[:, ACT_INDS, :, :], dim=1)
            # print(max_actuator, max_actuator.shape, max_actuator.dtype)
            self.world.mem[:, inds.last_move, :, :] = max_actuator / 2.0 # Normalize to 0-1 (HARDCODED, BE CAREFUL)

            # Directly update the world's communication channels
            self.world.mem[:, 3:, :, :] = torch.sigmoid(x[:, 3:, :, :])
            self.distribute_energy(self.world.mem, self.world.ti_indices)

            investments = x[:, INVEST_IDX, :, :] * self.world.mem[:, inds.energy, :, :]
            liquidations = x[:, LIQUIDATE_IDX, :, :] * self.world.mem[:, inds.infra, :, :]
            self.world.mem[:, inds.energy, :, :] += liquidations - investments
            self.world.mem[:, inds.infra, :, :] += investments - liquidations

            # self.apply_activations_to_world(self.world.mem, x, self.world.ti_indices)
            # self.world.stat("infra")

            return self.world.mem


    def perturb_weights(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data)
        # self.latent_conv.weight.data += perturbation_strength * torch.randn_like(self.latent_conv.weight.data)
        self.latent_conv_2.weight.data += perturbation_strength * torch.randn_like(self.latent_conv_2.weight.data)