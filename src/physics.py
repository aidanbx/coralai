import torch
import numpy as np
from typing import Tuple

def grow_muscle_csa(muscle_radii: torch.Tensor, radii_deltas: torch.Tensor,
                capital: torch.Tensor, growth_efficiency: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Update to work on batches of environments (simple)
    """
    Function to simulate the growth of muscle tissue. Modifies the radii and capital tensors in-place.

    Parameters:
    radii (torch.Tensor): The radii of the muscles of each cell. Shape: (kernel_size, width, height)
    radii_deltas (torch.Tensor): Desired changes in radii. Shape: (kernel_size, width, height)
    capital (torch.Tensor): capital present on each cell. Shape: (width, height)
    efficiency (torch.Tensor): The efficiency (heat loss) for a given change in muscle size. Shape: (width, height)

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: The new muscle radii and capital tensors.
    """
    zero_tensor = torch.tensor(0.0)
    csa_deltas = (muscle_radii + radii_deltas)**2 - muscle_radii**2  # cross-sectional area
    print(csa_deltas[[1,2]])

    negative_csa_deltas = torch.where(csa_deltas < 0, csa_deltas, zero_tensor)
    positive_csa_deltas = torch.where(csa_deltas > 0, csa_deltas, zero_tensor)

    # Atrophy muscle and convert to capital
    capital.sub_(torch.sum(negative_csa_deltas, dim=0) * growth_efficiency)

    new_csa_mags = muscle_radii**2.0
    new_csa_mags.add_(negative_csa_deltas)

    # Grow muscle from capital, if possible
    capital_desired = torch.sum(positive_csa_deltas, dim=0)
    capital_desired_safe = torch.where(capital_desired == 0, torch.tensor(1.0), capital_desired)
    csa_delta_distribution = positive_csa_deltas / capital_desired_safe.unsqueeze(0) 
    del capital_desired_safe # TODO: Free CUDA memory? Garbage Collect?  

    capital_consumed = torch.where(capital_desired > capital, capital, capital_desired)
    capital.sub_(capital_consumed)
    capital_consumed.mul_(growth_efficiency)
    new_csa_mags.addcmul_(capital_consumed.unsqueeze(0), csa_delta_distribution)
    torch.mul(torch.sqrt(new_csa_mags), torch.sign(muscle_radii + radii_deltas), out=muscle_radii)  

    return muscle_radii, capital


if __name__ == "__main__":
    # Tests if capital is consumed correctly and radii are updated in place

    import matplotlib.pyplot as plt

    rads_batch = torch.tensor([[-4]], dtype=torch.float32)
    rad_deltas_batch = torch.tensor([[2]], dtype=torch.float32)
    capital_batch = torch.tensor([3.0], dtype=torch.float32)
    efficiency_batch = torch.tensor([1.0], dtype=torch.float32)

    radii_results = []
    capital_results = []
    for _ in range(10):
        out = grow_muscle_csa(rads_batch, rad_deltas_batch, capital_batch, efficiency_batch)
        radii_results.append(out[0].item())
        capital_results.append(out[1].item())

    assert torch.isclose(rads_batch, torch.tensor([[4.358899]], dtype=torch.float32), atol=1e-6).all()
    assert torch.isclose(capital_batch, torch.tensor([0.0], dtype=torch.float32), atol=1e-6).all()

    plt.figure(figsize=(10, 5))
    plt.plot(radii_results, label='Radii')
    plt.plot(capital_results, label='capital')
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title('Growth of Muscle Tissue')
    plt.legend()
    plt.grid(True)
    plt.show()