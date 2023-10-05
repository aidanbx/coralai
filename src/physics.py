import torch
import numpy as np
from typing import Tuple
import importlib
import src.EINCASMConfig as EINCASMConfig
importlib.reload(EINCASMConfig)

# def generate_muscle_masks(cfg: EINCASMConfig, open_cells: torch.Tensor) -> torch.Tensor:
#     directional_masks = torch.ones((cfg.kernel.shape[0], *open_cells.shape), device=cfg.device, dtype=torch.bool)
#     for i in range(cfg.kernel.shape[0]):
#         directional_masks[i] = torch.roll(open_cells, shifts=tuple(map(int, cfg.kernel[i])), dims=[0, 1])
    
#     muscle_masks = directional_masks&open_cells
#     return muscle_masks

def grow_muscle_csa(cfg: EINCASMConfig, muscle_radii: torch.Tensor, radii_deltas: torch.Tensor,
                    capital: torch.Tensor, growth_efficiency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Update to work on batches of environments (simple)
    """
    Function to simulate the growth of muscle tissue. Modifies the radii and capital tensors in-place.

    Parameters:
    radii (torch.Tensor): The radii of the muscles of each cell. Shape: (num_muscles, width, height)
    radii_deltas (torch.Tensor): Desired changes in radii. Shape: (num_muscles, width, height)
    capital (torch.Tensor): capital present on each cell. Shape: (width, height)
    efficiency (torch.Tensor): The efficiency (heat loss) for a given change in muscle size. Shape: (width, height)

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: The new muscle radii and capital tensors.
    """
    # assert muscle_radii[~muscle_masks].sum() < 1e-6, "Muscle radii in/towards obstacle cells is not zero."
    # muscle_radii *= muscle_masks
    # radii_deltas *= muscle_masks
    # assert capital.min() >= 0, "Capital cannot be negative"
    # assert capital[~open_cells].sum() < 1e-6, "Capital in obstacle cells is not zero."
    # capital *= open_cells
    # assert growth_efficiency.min() >= 0, "Growth efficiency cannot be negative"

    csa_deltas = (muscle_radii + radii_deltas)**2 - muscle_radii**2  # cross-sectional area

    negative_csa_deltas = torch.where(csa_deltas < 0, csa_deltas, cfg.zero_tensor)
    positive_csa_deltas = torch.where(csa_deltas > 0, csa_deltas, cfg.zero_tensor)

    # Atrophy muscle and convert to capital
    new_csa_mags = muscle_radii**2.0
    # total_capital_before = capital.sum() + new_csa_mags.sum()
    new_csa_mags.add_(negative_csa_deltas)

    capital.sub_(torch.sum(negative_csa_deltas, dim=0).mul(growth_efficiency))

    # Grow muscle from capital, if possible
    total_csa_deltas = torch.sum(positive_csa_deltas, dim=0)
    total_csa_deltas_safe = torch.where(total_csa_deltas == 0, cfg.one_tensor, total_csa_deltas)
    csa_delta_distribution = positive_csa_deltas.div(total_csa_deltas_safe.unsqueeze(0))
    # del total_csa_deltas_safe # TODO: Free CUDA memory? Garbage Collect?  

    capital_consumed = torch.where(total_csa_deltas > capital, capital, total_csa_deltas)
    capital.sub_(capital_consumed)
    capital_consumed.mul_(growth_efficiency)
    # other than inefficiency, exchange rate between capital and muscle area is 1:1 
    new_csa_mags.addcmul_(capital_consumed.unsqueeze(0), csa_delta_distribution) 
    # This is allowed because even with no available capital a muscle may invert its polarity or go to 0 at only a cost of inefficiency
    torch.copysign(torch.sqrt(new_csa_mags), muscle_radii.add(radii_deltas), out=muscle_radii) 

    capital = torch.where(capital < 0.001, cfg.zero_tensor, capital)
    # assert capital.min() >= 0, "Oops, a cell's capital became negative during growth"
    # capital_diff = (capital.sum() + new_csa_mags.sum()) - total_capital_before
    # assert capital_diff <= 0, f"Oops, capital was invented during growth. Diff: {capital_diff}"

    return muscle_radii, capital


def activate_muscles(cfg: EINCASMConfig, muscle_radii: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    return (torch.sign(muscle_radii) * (muscle_radii ** 2)).mul(activations)    # Activation is a percentage (kinda) of CSA

def activate_and_mine_ports(cfg: EINCASMConfig, capital: torch.Tensor, port: torch.Tensor, mine_efficiency: torch.Tensor,
                            dispersal_rate: torch.Tensor, mine_muscle_radii: torch.Tensor, mine_activation: torch.Tensor,
                            regeneration_rate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        available = port * dispersal_rate
        extraction = activate_muscles(cfg, mine_muscle_radii, mine_activation)
        extracted = torch.clamp(extraction, -capital, available)
        capital_flow = torch.where(extracted > 0, extracted * mine_efficiency, extracted)
        port_flow = torch.where(extracted < 0, extracted * mine_efficiency, extracted)
        
        capital += capital_flow
        port -= port_flow

        port += regeneration_rate

        return port, capital


def activate_flow_muscles(cfg: EINCASMConfig, capital: torch.Tensor, waste: torch.Tensor, flow_muscle_radii: torch.Tensor,
                              flow_activations: torch.Tensor, flow_efficiency: torch.Tensor,
                              obstacles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to activate muscles and simulate flow of cytoplasm. Modifies the capital tensor in-place.

    Parameters:
    capital (torch.Tensor): The capital present in each cell. Shape: (width, height)
    muscle_radii (torch.Tensor): The radii of the muscles of each cell. Shape: (kernel_size, width, height)
    activations (torch.Tensor): The activation levels of the muscles 0-1. Shape: (width, height)
    flow_efficiency (torch.Tensor): The efficiency of flow. Shape: (width, height)
    kernel (torch.Tensor): The kernel for exchanging capital. Shape: (kernel_size, num_dims)

    Returns:
    torch.Tensor: The updated capital tensor.
    """
    total_capital_before = capital.sum()
    assert capital.min() >= 0, "Capital cannot be negative"

    cfg.one_tensor = torch.tensor([1.0], device=cfg.device, dtype=cfg.float_dtype)

    # Enforce obstacles
    flow_activations *= (1-obstacles)

    # Invert negative flows across kernel
    roll_shift = (len(cfg.kernel)-1)//2
    flows = activate_muscles(cfg, flow_muscle_radii, flow_activations)
    positive_flows = torch.where(flows[1:] > 0, flows[1:], cfg.zero_tensor)
    negative_flows = torch.where(flows[1:] < 0, flows[1:], cfg.zero_tensor)
    flows[1:] = positive_flows.sub(torch.roll(negative_flows, roll_shift, dims=0))
    flows[0] = torch.abs(flows[0]) # Self flow is always positive
    del positive_flows, negative_flows

    # Distribute cytoplasm AND WASTE across flows
    total_flow = torch.sum(flows, dim=0)
    total_flow_desired_safe = torch.where(total_flow == 0, cfg.one_tensor, total_flow)
    flow_distribution = flows.div(total_flow_desired_safe.unsqueeze(0))
    del total_flow_desired_safe

    max_flow_possible = torch.min(capital + waste, capital.mul(1-flow_efficiency))
    total_flow = torch.where(total_flow>max_flow_possible,max_flow_possible,total_flow)
    capital.sub_(total_flow.mul(1-flow_efficiency))  # enforce cost of flow before distributing

    mass = capital + waste
    percent_waste = waste.div(torch.where(mass == 0, cfg.one_tensor, mass))
    del mass
    waste_out = percent_waste * total_flow
    capital_out = total_flow - waste_out
    capital.sub_(capital_out)
    waste.sub_(waste_out)

    capital_flows = flow_distribution.mul(capital_out.unsqueeze(0))
    waste_flows = flow_distribution.mul(waste_out.unsqueeze(0))
    del waste_out, capital_out

    received_capital = torch.sum(
                            torch.stack(
                                [torch.roll(capital_flows[i], shifts=tuple(map(int, cfg.kernel[i])), dims=[0, 1])
                                for i in range(cfg.kernel.shape[0])]
                            ), dim=0
                        )
    received_waste = torch.sum(
                            torch.stack(
                                [torch.roll(waste_flows[i], shifts=tuple(map(int, cfg.kernel[i])), dims=[0, 1])
                                for i in range(cfg.kernel.shape[0])]
                            ), dim=0
                        )
    del capital_flows, waste_flows
    capital.add_(received_capital)
    waste.add_(received_waste)
    del received_capital, received_waste

    capital = torch.where(capital < 0.01, cfg.zero_tensor, capital)

    return capital, waste


# def activate_mine_muscle(cfg: EINCASMConfig, mine_activations: torch.Tensor, muscle_radii: torch.Tensor,
#                          capital: torch.Tensor, waste: torch.Tensor, obstacles: torch.Tensor)
          

# def run_physics(cfg: EINCASMConfig, env_channels: torch.Tensor, live_channels: torch.Tensor) -> torch.Tensor:
    