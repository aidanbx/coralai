import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm


def activate_outputs(substrate, ind_of_middle):
    inds = substrate.ti_indices[None]
    substrate.mem[:, inds.com] = torch.sigmoid(ch_norm(substrate.mem[:, inds.com]))
    substrate.mem[:, [inds.acts_invest, inds.acts_liquidate]] = torch.sigmoid(ch_norm(substrate.mem[:, [inds.acts_invest, inds.acts_liquidate]]))

    # substrate.mem[:, inds.acts_explore] = nn.ReLU()(substrate.mem[:, inds.acts_explore])
    # mean_activation = torch.mean(substrate.mem[0, inds.acts_explore], dim=0)
    # substrate.mem[0, inds.acts_explore[ind_of_middle]] = mean_activation + 2
    substrate.mem[0, inds.acts_explore] = torch.softmax(substrate.mem[0, inds.acts_explore], dim=1)

    substrate.mem[0, inds.acts] = torch.where(substrate.mem[0, inds.genome] < 0, 0, substrate.mem[0, inds.acts])

@ti.kernel
def flow_energy_down(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                     max_energy: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        if central_energy > max_energy:
            energy_sum_inverse = 0.0
            # Calculate the sum of the inverse of energystructure levels for neighboring cells
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                energy_level = mem[0, inds.energy, neigh_x, neigh_y]
                # Avoid division by zero by ensuring a minimum energystructure level
                energy_level = max(energy_level, 0.0001)  # Assuming 0.0001 as a minimum to avoid division by zero
                energy_sum_inverse += 1.0 / energy_level
            # Distribute energy based on the inverse proportion of energystructure levels
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                neigh_energy = mem[0, inds.energy, neigh_x, neigh_y]
                neigh_energy = max(neigh_energy, 0.0001)  # Again, ensuring a minimum energystructure level
                # Calculate the share of energy based on the inverse of energystructure level
                energy_share = central_energy * ((1.0 / neigh_energy) / energy_sum_inverse)
                out_energy_mem[neigh_x, neigh_y] += energy_share
        else:
            out_energy_mem[i, j] += mem[0, inds.energy, i, j]

@ti.kernel
def distribute_energy(mem: ti.types.ndarray(), out_energy: ti.types.ndarray(), max_energy: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        if mem[0, inds.energy, i, j] > max_energy:
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                out_energy[neigh_x, neigh_y] += (mem[0, inds.energy, i, j] / kernel.shape[0])
        else:
            out_energy[i, j] += mem[0, inds.energy, i, j]

@ti.kernel
def flow_energy_up(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                      kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        infra_sum = 0.0
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            infra_sum += mem[0, inds.infra, neigh_x, neigh_y]
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            neigh_infra = mem[0, inds.infra, neigh_x, neigh_y]
            out_energy_mem[neigh_x, neigh_y] += (
                central_energy * ((neigh_infra/infra_sum)))
            
@ti.kernel
def distribute_infra(mem: ti.types.ndarray(), out_infra: ti.types.ndarray(), max_infra: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        if mem[0, inds.infra, i, j] > max_infra:
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                out_infra[neigh_x, neigh_y] += (mem[0, inds.infra, i, j] / kernel.shape[0])
        else:
            out_infra[i, j] += mem[0, inds.infra, i, j]
    

def energy_physics(substrate, kernel, max_infra, max_energy):
    inds = substrate.ti_indices[None]
    # substrate.mem[0, inds.infra] = torch.clamp(substrate.mem[0, inds.infra], 0.0001, 100)

    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    flow_energy_up(substrate.mem, energy_out_mem, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = energy_out_mem

    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    distribute_energy(substrate.mem, energy_out_mem, max_energy, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = energy_out_mem

    infra_out_mem = torch.zeros_like(substrate.mem[0, inds.infra])
    distribute_infra(substrate.mem, infra_out_mem, max_infra, kernel, substrate.ti_indices)
    substrate.mem[0, inds.infra] = infra_out_mem


def invest_liquidate(substrate):
    inds = substrate.ti_indices[None]
    investments = substrate.mem[0, inds.acts_invest] * substrate.mem[0, inds.energy]
    liquidations = substrate.mem[0, inds.acts_liquidate] * substrate.mem[0, inds.infra]
    substrate.mem[0, inds.energy] += liquidations - investments
    substrate.mem[0, inds.infra] += investments - liquidations


@ti.kernel
def explore(mem: ti.types.ndarray(), max_act_i: ti.types.ndarray(),
            infra_delta: ti.types.ndarray(), winning_genomes: ti.types.ndarray(),
            kernel: ti.types.ndarray(), explore_inds: ti.types.ndarray(), ti_inds: ti.template()):
    # incoming investments
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        winning_genome = mem[0, inds.genome, i, j]
        max_bid = 0.0
        for offset_n in ti.ndrange(kernel.shape[0]):
            offset_x = kernel[offset_n, 0]
            offset_y = kernel[offset_n, 1]
            neigh_x = (i + offset_x) % mem.shape[2]
            neigh_y = (j + offset_y) % mem.shape[3]
            if mem[0, inds.genome, neigh_x, neigh_y] < 0:
                continue
            neigh_max_act_i = max_act_i[neigh_x, neigh_y]
            bid = 0.0
            # if neigh's explore dir points towards this center
            if (((kernel[neigh_max_act_i, 0] + offset_x) == 0 and (kernel[neigh_max_act_i, 1] + offset_y) == 0) or
                (offset_x == 0 and offset_y == 0)): # central cell always gets a bid 
                if offset_x == 0 and offset_y == 0:
                    bid = mem[0, inds.infra, i, j] * 0.5 # max possible bid
                else:
                    bid = mem[0, inds.infra, neigh_x, neigh_y] * 0.3
                    infra_delta[neigh_x, neigh_y] -= bid # bids are always taken as investment
                    infra_delta[i, j] += bid * 0.9
                if bid > max_bid:
                    max_bid = bid
                    winning_genome = mem[0, inds.genome, neigh_x, neigh_y]
        winning_genomes[i, j] = winning_genome
        # if max_bid > mem[0, inds.infra, i, j]/2 or mem[0, inds.genome, i, j] < 0:
        # else:
        #     winning_genomes[i, j] = mem[0, inds.genome, i, j]


def explore_physics(substrate, kernel):
    inds = substrate.ti_indices[None]

    max_act_i = torch.argmax(substrate.mem[0, inds.acts_explore], dim=0) # be warned, this is the index of the actuator not the index in memory, so 0-6 not
    infra_delta = torch.zeros_like(substrate.mem[0, inds.infra])
    winning_genome = torch.zeros_like(substrate.mem[0, inds.genome])
    # This is intermediate storage for each cell in the kernel to use:
    explore_inds = torch.tensor(inds.acts_explore)
    explore(substrate.mem, max_act_i,
            infra_delta, winning_genome,
            kernel, explore_inds, substrate.ti_indices)
    # handle_investment(substrate, infra_delta)
    substrate.mem[0, inds.genome] = winning_genome
    substrate.mem[0, inds.infra] += infra_delta
