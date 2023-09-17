import torch
import numpy as np
from typing import Tuple

def grow_muscle(radii: torch.Tensor, radii_deltas: torch.Tensor,
                capital: torch.Tensor, efficiency: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to simulate the growth of muscle tissue. Modifies the radii and capital tensors in-place.

    Parameters:
    radii (torch.Tensor): The radii of the muscles of each cell. Shape: (batch_size, num_muscles)
    radii_deltas (torch.Tensor): Desired changes in radii. Shape: (batch_size, num_muscles)
    capital (torch.Tensor): capital present on each cell. Shape: (batch_size,)
    efficiency (torch.Tensor): The efficiency (heat loss) for a given change in muscle size. Shape: (batch_size,)

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: The new muscle radii and capital for
      each cell. Shape: ((batch_size, num_muscles), (batch_size,))
    """
    zero_tensor = torch.tensor(0.0)
    csa_deltas = (radii + radii_deltas)**2 - radii**2  # cross-sectional area

    negative_csa_deltas = torch.where(csa_deltas < 0, csa_deltas, zero_tensor)
    positive_csa_deltas = torch.where(csa_deltas > 0, csa_deltas, zero_tensor)

    # Atrophy muscle and convert to capital
    capital.sub_(torch.sum(negative_csa_deltas, dim=1) * efficiency)
    
    new_csa_mags = radii**2.0
    new_csa_mags.add_(negative_csa_deltas)

    # Grow muscle from capital, if possible
    capital_desired = torch.sum(positive_csa_deltas, dim=1)
    capital_desired_safe = torch.where(capital_desired == 0, torch.tensor(1.0), capital_desired)
    csa_delta_distribution = positive_csa_deltas / capital_desired_safe.unsqueeze(1)

    capital_consumed = torch.where(capital_desired > capital, capital, capital_desired)
    capital.sub_(capital_consumed)
    capital_consumed.mul_(efficiency)
    new_csa_mags.addcmul_(capital_consumed.unsqueeze(1), csa_delta_distribution)
    torch.mul(torch.sqrt(new_csa_mags), torch.sign(radii + radii_deltas), out=radii)  

    return radii, capital


def grow_old(rads_b, rad_deltas_b, capital_b, efficiency_b):
    csa_deltas_b = (rads_b + rad_deltas_b)**2 - rads_b**2 # cross-sectional area

    # Atrophy muscle and convert to capital
    capital_b -= torch.sum(torch.where(csa_deltas_b < 0, csa_deltas_b, torch.tensor(0.0)),dim=1) * efficiency_b

    new_csa_mags_b = rads_b**2.0
    new_csa_mags_b[csa_deltas_b < 0] += csa_deltas_b[csa_deltas_b < 0]

    # Grow myscle from capital, if possible
    capital_desired_b = torch.sum(torch.where(csa_deltas_b > 0, csa_deltas_b, torch.tensor(0.0)),dim=1)
    csa_delta_distribution_b = torch.where(csa_deltas_b > 0, csa_deltas_b, torch.tensor(0.0)) / capital_desired_b.unsqueeze(1)

    capital_consumed_b = torch.where(capital_desired_b > capital_b, capital_b, capital_desired_b)
    csa_grown_b = capital_consumed_b * efficiency_b
    new_csa_mags_b = torch.where(csa_deltas_b > 0, new_csa_mags_b + csa_grown_b.unsqueeze(1) * csa_delta_distribution_b, new_csa_mags_b)

    capital_b -= capital_consumed_b

    new_rad_mags_b = torch.sqrt(new_csa_mags_b)
    new_signs = torch.sign(rads_b + rad_deltas_b)

    return new_rad_mags_b * new_signs, capital_b


def grow_dumb(rads, rad_deltas, capital, efficiency):
    activate_muscle_growth = np.vectorize(lambda rads, rad_deltas: (rads + rad_deltas)**2 - rads**2)
    csa_deltas = activate_muscle_growth(rads, rad_deltas) # cross-sectional area
    positive_csa_deltas = csa_deltas[csa_deltas > 0]
    negative_csa_deltas = csa_deltas[csa_deltas < 0]

    # Atrophy muscle and convert to capital
    capital -= sum(negative_csa_deltas) * efficiency
    new_csa_mags = rads**2.0
    new_csa_mags[csa_deltas < 0] += negative_csa_deltas

    # Grow muscle from capital, if possible
    capital_desired = sum(positive_csa_deltas) # before efficiency loss
    csa_delta_distribution = positive_csa_deltas / capital_desired

    if capital_desired > capital:
        capital_consumed = capital
    else:
        capital_consumed = capital_desired

    csa_grown = capital_consumed * efficiency
    new_csa_mags[csa_deltas > 0] += csa_grown * csa_delta_distribution

    capital -= capital_consumed

    new_rad_mags = np.sqrt(new_csa_mags)
    new_signs = np.sign(rads + rad_deltas)


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
        out = grow_muscle(rads_batch, rad_deltas_batch, capital_batch, efficiency_batch)
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