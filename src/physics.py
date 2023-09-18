import torch
import numpy as np
from typing import Tuple

def activate_muscles_and_flow(capital: torch.Tensor, muscle_radii: torch.Tensor,
                              activations: torch.Tensor, flow_efficiency: torch.Tensor,
                              kernel: torch.Tensor) -> torch.Tensor:
    """
    Function to activate muscles and simulate flow of cytoplasm. Modifies the capital tensor in-place.

    Parameters:
    capital (torch.Tensor): The capital present in each cell. Shape: (width, height)
    muscle_radii (torch.Tensor): The radii of the muscles of each cell. Shape: (kernel_size, width, height)
    activations (torch.Tensor): The activation levels of the muscles. Shape: (width, height)
    flow_efficiency (torch.Tensor): The efficiency of flow. Shape: (width, height)
    kernel (torch.Tensor): The kernel for exchanging capital. Shape: (kernel_size, num_dims)

    Returns:
    torch.Tensor: The updated capital tensor.
    """
    total_capital_before = capital.sum()
    assert capital.min() >= 0, "Capital cannot be negative"

    # Invert negative flows across kernel
    roll_shift = (len(kernel)-1)//2
    zero_tensor = torch.tensor([0.0])
    flows = (torch.sign(muscle_radii) * (muscle_radii ** 2)).mul(activations)    # Activation is a percentage (kinda) of CSA
    positive_flows = torch.where(flows[1:] > 0, flows[1:], zero_tensor)
    negative_flows = torch.where(flows[1:] < 0, flows[1:], zero_tensor)
    flows[1:] = positive_flows.sub(torch.roll(negative_flows, roll_shift, dims=0))
    flows[0] = torch.abs(flows[0])
    del positive_flows, negative_flows

    # Distribute cytoplasm across flows
    total_flow_desired = torch.sum(flows, dim=0)
    total_flow_desired_safe = torch.where(total_flow_desired == 0, torch.tensor(1.0), total_flow_desired)
    flow_distribution = flows.div(total_flow_desired_safe.unsqueeze(0))
    del total_flow_desired_safe

    total_capital_outflow = torch.where(total_flow_desired > capital, capital, total_flow_desired)
    capital.sub_(total_capital_outflow)
    assert capital.min() >= 0, "Capital cannot be negative"
    total_capital_outflow.mul_(flow_efficiency)
    torch.mul(total_capital_outflow.unsqueeze(0), flow_distribution, out=flows)

    # Exchange capital according to the kernel
    received_capital = torch.sum(
                            torch.stack(
                                [torch.roll(flows[i], shifts=tuple(map(int, kernel[i])), dims=[0, 1])
                                for i in range(kernel.shape[0])]
                            ), dim=0
                        )
    capital.add_(received_capital)
    del received_capital

    capital = torch.where(capital < 0.001, torch.tensor(0.0), capital)
    assert total_capital_before >= capital.sum(), "Capital must be lost in the system"

    return capital


def grow_muscle_csa(muscle_radii: torch.Tensor, radii_deltas: torch.Tensor,
                capital: torch.Tensor, growth_efficiency: torch.Tensor,
                muscle_masks: torch.Tensor, open_cells: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Update to work on batches of environments (simple)
    """
    Function to simulate the growth of muscle tissue. Modifies the radii and capital tensors in-place.

    Parameters:
    radii (torch.Tensor): The radii of the muscles of each cell. Shape: (kernel_size, width, height)
    radii_deltas (torch.Tensor): Desired changes in radii. Shape: (kernel_size, width, height)
    capital (torch.Tensor): capital present on each cell. Shape: (width, height)
    efficiency (torch.Tensor): The efficiency (heat loss) for a given change in muscle size. Shape: (width, height)
    obstacle_masks (torch.Tensor): The directions for each cell that lead to an obstacle. Shape: (kernel_size, width, height)
    open_cells (torch.Tensor): The cells that are not obstacles. Shape: (width, height)
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor]: The new muscle radii and capital tensors.
    """
    muscle_radii *= muscle_masks
    radii_deltas *= muscle_masks
    assert capital[~open_cells].sum() < 1e-6, "Capital in obstacle cells is not zero."
    capital *= open_cells

    zero_tensor = torch.tensor(0.0)
    csa_deltas = (muscle_radii + radii_deltas)**2 - muscle_radii**2  # cross-sectional area

    negative_csa_deltas = torch.where(csa_deltas < 0, csa_deltas, zero_tensor)
    positive_csa_deltas = torch.where(csa_deltas > 0, csa_deltas, zero_tensor)

    # Atrophy muscle and convert to capital
    capital.sub_(torch.sum(negative_csa_deltas, dim=0).mul(growth_efficiency))

    new_csa_mags = muscle_radii**2.0
    new_csa_mags.add_(negative_csa_deltas)

    # Grow muscle from capital, if possible
    total_csa_deltas = torch.sum(positive_csa_deltas, dim=0)
    total_csa_deltas_safe = torch.where(total_csa_deltas == 0, torch.tensor(1.0), total_csa_deltas)
    csa_delta_distribution = positive_csa_deltas.div(total_csa_deltas_safe.unsqueeze(0))
    del total_csa_deltas_safe # TODO: Free CUDA memory? Garbage Collect?  

    capital_consumed = torch.where(total_csa_deltas > capital, capital, total_csa_deltas)
    capital.sub_(capital_consumed)
    capital_consumed.mul_(growth_efficiency)
    new_csa_mags.addcmul_(capital_consumed.unsqueeze(0), csa_delta_distribution)
    torch.mul(torch.sqrt(new_csa_mags), torch.sign(muscle_radii.add(radii_deltas)), out=muscle_radii)  

    return muscle_radii, capital