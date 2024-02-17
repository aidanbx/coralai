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
            max_infra = mem[0, inds.infra, i, j]
            max_i, max_j = i, j
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                ni, nj = (i + di) % self.world.w, (j + dj) % self.world.h
                if mem[0, inds.infra, ni, nj] > max_infra:
                    max_infra = mem[0, inds.infra, ni, nj]
                    max_i, max_j = ni, nj
            if max_i != i or max_j != j:
                mem[0, inds.energy, max_i, max_j] += mem[0, inds.energy, i, j]
                mem[0, inds.energy, i, j] = 0.0


    @ti.kernel
    def explore(self, mem: ti.types.ndarray(), x: ti.types.ndarray(), ti_inds: ti.template()):
        """
        Explore takes the free energy on the current cell and distributed a proportion of it as determined by the cell's activation to its neighbors, as determined by a moore's neighborhood
        """
        inds = ti_inds[None]
        for i, j in ti.ndrange(self.world.w, self.world.h):
            energy = mem[0, inds.energy, i, j]
            explore_act = x[0, EXPLORE_IDX, i, j]
            distributed_energy = (explore_act * energy)/8.0
            # Apply explore operation
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):  # Moore's neighborhood
                if di == 0 and dj == 0:
                    continue  # Skip the current cell
                ni, nj = (i + di) % self.world.w, (j + dj) % self.world.h
                mem[0, inds.infra, ni, nj] += distributed_energy
            mem[0, inds.energy, i, j] -= distributed_energy * 8.0


    def forward(self, x):
        inds = self.world.ti_indices[None]
        with torch.no_grad():
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = self.latent_conv_2(x)

            # Directly update the world's communication channels
            self.world.mem[:, 3:, :, :] = torch.sigmoid(x[:, 3:, :, :])
            # self.world.mem[:, inds.infra, :, :] -= 0.2
            # # ensure energy is non-negative
            # self.world.mem[:, inds.infra, :, :] = torch.clamp(self.world.mem[:, inds.infra, :, :], min=0.0)

            self.distribute_energy(self.world.mem, self.world.ti_indices)
            x[:, ACT_INDS, :, :] = torch.softmax(x[:, ACT_INDS, :, :], dim=1)

            investments = x[:, INVEST_IDX, :, :] * self.world.mem[:, inds.energy, :, :]
            liquidations = x[:, LIQUIDATE_IDX, :, :] * self.world.mem[:, inds.infra, :, :]
            assert (investments >= 0).all()
            assert (liquidations >= 0).all()
            self.world.mem[:, inds.energy, :, :] += liquidations - investments
            self.world.mem[:, inds.infra, :, :] += investments - liquidations
            assert (self.world.mem[:, inds.energy, :, :] >= 0).all()
            assert (self.world.mem[:, inds.infra, :, :] >= 0).all()

            self.explore(self.world.mem, x, self.world.ti_indices)
            assert (self.world.mem[:, inds.infra, :, :] >= 0).all()
            assert (self.world.mem[:, inds.energy, :, :] >= 0).all()

            # max_actuator = torch.argmax(x[:, ACT_INDS, :, :], dim=1)
            # self.world.mem[:, inds.last_move, :, :] = max_actuator / 2.0 # Normalize to 0-1 (HARDCODED, BE CAREFUL)

            # self.world.mem[:, [inds.energy, inds.infra], :, :] = ch_norm(self.world.mem[:, [inds.energy, inds.infra], :, :])
            # self.world.mem[:, [inds.energy, inds.infra], :, :] = torch.sigmoid(self.world.mem[:, [inds.energy, inds.infra], :, :])
            return self.world.mem


    def perturb_weights(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data)
        # self.latent_conv.weight.data += perturbation_strength * torch.randn_like(self.latent_conv.weight.data)
        self.latent_conv_2.weight.data += perturbation_strength * torch.randn_like(self.latent_conv_2.weight.data)