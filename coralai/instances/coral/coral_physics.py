import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm


@ti.kernel
def distribute_energy(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                      kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        infra_softmax_denominator = 0.0
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            infra_softmax_denominator += ti.math.pow(ti.math.e, mem[0, inds.infra, neigh_x, neigh_y])
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            neigh_infra = mem[0, inds.infra, neigh_x, neigh_y]
            out_energy_mem[neigh_x, neigh_y] += (
                central_energy * (ti.math.pow(ti.math.e, neigh_infra)/infra_softmax_denominator))


def invest_liquidate(substrate):
    inds = substrate.ti_indices[None]
    substrate.mem[0, [inds.invest_act, inds.liquidate_act]] = torch.softmax(substrate.mem[0, [inds.invest_act, inds.liquidate_act]], dim=0)
    investments = substrate.mem[0, inds.acts_investment] * substrate.mem[0, inds.energy]
    liquidations = substrate.mem[0, inds.acts_liquidate] * substrate.mem[0, inds.infra]
    substrate.mem[0, inds.energy] += liquidations - investments
    substrate.mem[0, inds.infra] += investments - liquidations


def decay_add_energy(substrate):
    pass
    # inds = substrate.ti_indices[None]
    # # Energy decay and random energy input
    # normal_rand = torch.randn_like(substrate.mem[0, inds.energy])
    # normal_rand = normal_rand * normal_rand
    # substrate.mem[0, inds.energy] += normal_rand * 0.01
    # normal_rand = torch.randn_like(substrate.mem[0, inds.energy])
    # normal_rand = normal_rand * normal_rand
    # substrate.mem[0, inds.energy] -= normal_rand * 0.1
    # substrate.mem[0, inds.infra] -= normal_rand * 0.1
    # substrate.mem[0, inds.energy] = torch.clamp(substrate.mem[0, inds.energy], 0, 99)
    # substrate.mem[0, inds.infra] = torch.clamp(substrate.mem[0, inds.infra], 0, 99)


@ti.kernel
def explore(mem: ti.types.ndarray(), investments: ti.types.ndarray(), 
            incoming_genomes: ti.types.ndarray(), max_act_i: ti.types.ndarray(),
            genome_bids: ti.types.ndarray(), winning_bid: ti.types.ndarray(),
            kernel: ti.types.ndarray(), ti_inds: ti.template()):
    # incoming investments
    inds = ti_inds[None]
    for center_x, center_y in ti.ndrange(mem.shape[2], mem.shape[3]):
        winning_bid_neigh_num = 999999
        max_bid = -1
        for offset_n in ti.ndrange(kernel.shape[0]):
            offset_x = kernel[offset_n, 0]
            offset_y = kernel[offset_n, 1]
            neigh_x = (center_x + offset_x) % mem.shape[2]
            neigh_y = (center_y + offset_y) % mem.shape[3]
            neigh_max_act_i = max_act_i[neigh_x, neigh_y]

            genome_bids[offset_n, 0] = mem[0, inds.genome, center_x, center_y] # central genome (in case no investments)
            genome_bids[offset_n, 1] = 0 # bid val
            # if neigh's explore dir points towards this center
            if (((kernel[neigh_max_act_i, 0] + offset_x) == 0 and (kernel[neigh_max_act_i, 1] + offset_y) == 0) or
                (offset_x == 0 and offset_y == 0)): # central cell always gets a bid 
                if offset_x == 0 and offset_y == 0:
                    neigh_explore_bid = mem[0, inds.infra, center_x, center_y] # max possible bid
                else:
                    neigh_explore_bid = (
                        mem[0, inds.energy, neigh_x, neigh_y] *
                        mem[0, inds.acts_explore[neigh_max_act_i], neigh_x, neigh_y])
                    investments[neigh_x, neigh_y] -= neigh_explore_bid # bids are always taken as investment
                    investments[center_x,center_y] += neigh_explore_bid
                neigh_genome = mem[0, inds.genome, neigh_x, neigh_y] # could be central cell
                investment_stored_in = offset_n
                for prev_investment_k in range(offset_n): # Add up investments from the same genome
                    if genome_bids[prev_investment_k, 0] == neigh_genome:
                        genome_bids[prev_investment_k, 1] += neigh_explore_bid # count the investment as its neighbor's
                        investment_stored_in = prev_investment_k
                if investment_stored_in == offset_n: # no investments from the same genome
                    genome_bids[offset_n, 0] = neigh_genome
                    genome_bids[offset_n, 1] = neigh_explore_bid
                if genome_bids[investment_stored_in, 1] > max_bid:
                    max_bid = genome_bids[investment_stored_in, 1]
                    winning_bid_neigh_num = investment_stored_in
        incoming_genomes[center_x, center_y] = genome_bids[winning_bid_neigh_num, 0]
        winning_bid[center_x, center_y] = genome_bids[winning_bid_neigh_num, 1]


def handle_investment(substrate, investments):
    inds = substrate.ti_indices[None]
    # remove energy where investments are negative, convert investments to infra where positive
    substrate.mem[0, inds.energy] = torch.where(investments < 0, substrate.mem[0, inds.energy] - investments, substrate.mem[0, inds.energy])
    assert torch.all(substrate.mem[0, inds.energy] >= 0)
    substrate.mem[0, inds.infra] += torch.where(investments > 0, substrate.mem[0, inds.infra] + torch.log(investments+1), substrate.mem[0, inds.infra])


def get_merging_cells(substrate, new_genomes, winning_bid):
    inds = substrate.ti_indices[None]
    bid_ratios = substrate.mem[0, inds.infra] / winning_bid
    substrate.mem[0, inds.genome] = torch.where(bid_ratios > 1.3, new_genomes, substrate.mem[0, inds.genome])
    # get coords where bid ratio is between 0.7 and 1.3
    return torch.where(bid_ratios < 1.3 and bid_ratios > 0.7)


def explore_physics(substrate, kernel):
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.explore_acts] = torch.softmax(substrate.mem[0, inds.explore_acts], dim=1)
    max_act_i = torch.argmax(substrate.mem[0, inds.explore_acts], dim=0) # be warned, this is the index of the actuator not the index in memory, so 0-6 not
    investments = torch.zeros_like(substrate.mem[0, inds.infra])
    incoming_genome_matrix = torch.full_like(substrate.mem[0, inds.genome], -1)
    # This is intermediate storage for each cell in the kernel to use:
    genome_bids = torch.zeros((kernel.shape[0], 2), device=substrate.torch_device) # [[genome, bid_val],[..]]
    winning_bid = torch.zeros_like(substrate.mem[0, inds.infra])
    explore(substrate.mem, investments,
            incoming_genome_matrix, max_act_i,
            genome_bids, winning_bid,
            kernel, substrate.ti_indices)
    merging_cell_coords = get_merging_cells(substrate, incoming_genome_matrix, winning_bid)
    handle_investment(substrate, investments)
    return merging_cell_coords, incoming_genome_matrix


def energy_physics(substrate, kernel):
    inds = substrate.ti_indices[None]
    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    distribute_energy(substrate.mem, energy_out_mem, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = energy_out_mem


def apply_physics(substrate, ecosystem, kernel):
    inds = substrate.ti_indices[None]

    substrate.mem[0, inds.com] = torch.sigmoid(nn.ReLU()(ch_norm(substrate.mem[:, inds.com])))
    # invest_liquidate(substrate)
    # merging_cell_coords, incoming_genome_matrix = explore_physics(substrate, kernel)
    # ecosystem.sexual_reproduction(merging_cell_coords, incoming_genome_matrix)
    # energy_physics(substrate, kernel)
