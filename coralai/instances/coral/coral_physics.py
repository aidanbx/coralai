import torch
import taichi as ti
import torch.nn as nn

from ...dynamics.nn_lib import ch_norm
from ...dynamics.Physics import Physics


LIQUIDATE_IDX = 0
INVEST_IDX = 1
EXPLORE_IDX = 2
ACT_INDS = [LIQUIDATE_IDX, INVEST_IDX, EXPLORE_IDX]

@ti.data_oriented
class CoralPhysics(Physics):
    def __init__(self):
        super(CoralPhysics, self).__init__()
        pass


    @ti.kernel
    def distribute_energy(self, mem: ti.types.ndarray(), ti_inds: ti.template()):
        """
        For every given cell, transfer all of its energy to its neighbor with the highest infrastructure value
        
        Moore's neighborhood, including central cell
        Torus boundary condition
        """
        inds = ti_inds[None]
        for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
            max_infra = mem[0, inds.infra, i, j]
            max_i, max_j = i, j
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                ni, nj = (i + di) % mem.shape[2], (j + dj) % mem.shape[3]
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
        for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
            if x[0, EXPLORE_IDX, i, j] > ti.math.max(x[0, INVEST_IDX, i, j], x[0, LIQUIDATE_IDX, i, j]):
                distributed_energy = x[0, EXPLORE_IDX, i, j] * mem[0, inds.energy, i, j] / 8.0
                # Apply explore operation
                for di, dj in ti.ndrange((-1, 2), (-1, 2)):  # Moore's neighborhood
                    if di == 0 and dj == 0:
                        continue  # Skip the current cell
                    ni, nj = (i + di) % mem.shape[2], (j + dj) % mem.shape[3]
                    mem[0, inds.infra, ni, nj] += distributed_energy  # Distribute equally to neighbors, with a 5% loss
                mem[0, inds.energy, i, j] -= distributed_energy * 8.0  # Remove the energy from the current cell


    def decay_add_energy(self, world):
        inds = world.ti_indices[None]
        # Energy decay and random energy input
        world.mem[:, inds.energy] = world.mem[:, inds.energy] * 0.99 + torch.rand_like(world.mem[:, inds.energy]) * 0.01
        world.mem[:, inds.infra] = world.mem[:, inds.infra] * 0.99  # Infrastructure decay


    def apply_actuators(self, world, actuator_values):
        inds = world.ti_indices[None]
        self.decay_add_energy(world)
        actuator_values[:, ACT_INDS, :, :] = torch.softmax(actuator_values[:, ACT_INDS, :, :], dim=1)

        max_actuator = torch.argmax(actuator_values[:, ACT_INDS, :, :], dim=1)
        world.mem[:, inds.last_move, :, :] = max_actuator / 2.0 # Normalize to 0-1 (HARDCODED, BE CAREFUL)

        world.mem[:, 3:, :, :] = torch.sigmoid(nn.ReLU()(ch_norm(actuator_values[:, 3:, :, :])))  # Update the communication channels

        investments = actuator_values[:, INVEST_IDX, :, :] * world.mem[:, inds.energy, :, :]
        liquidations = actuator_values[:, LIQUIDATE_IDX, :, :] * world.mem[:, inds.infra, :, :]
        world.mem[:, inds.energy, :, :] += liquidations - investments
        world.mem[:, inds.infra, :, :] += investments - liquidations

        self.explore(world.mem, actuator_values, world.ti_indices)
        self.distribute_energy(world.mem, world.ti_indices)
