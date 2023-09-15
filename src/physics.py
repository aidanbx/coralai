import torch
import numpy as np
from typing import Tuple

def grow_muscle(radii: torch.Tensor, radii_deltas: torch.Tensor,
                cytoplasm: torch.Tensor, efficiency: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to simulate the growth of muscle tissue. Modifies the radii and cytoplasm tensors in-place.

    Parameters:
    radii (torch.Tensor): The radii of the muscles of each cell. Shape: (batch_size, num_muscles)
    radii_deltas (torch.Tensor): Desired changes in radii. Shape: (batch_size, num_muscles)
    cytoplasm (torch.Tensor): Cytoplasm present on each cell. Shape: (batch_size,)
    efficiency (torch.Tensor): The efficiency (heat loss) for a given change in muscle size. Shape: (batch_size,)

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: The new muscle radii and cytoplasm for
      each cell. Shape: ((batch_size, num_muscles), (batch_size,))
    """
    zero_tensor = torch.tensor(0.0)
    csa_deltas = (radii + radii_deltas)**2 - radii**2  # cross-sectional area

    negative_csa_deltas = torch.where(csa_deltas < 0, csa_deltas, zero_tensor)
    positive_csa_deltas = torch.where(csa_deltas > 0, csa_deltas, zero_tensor)

    # Atrophy muscle and convert to cyt
    cytoplasm.sub_(torch.sum(negative_csa_deltas, dim=1) * efficiency)
    
    new_csa_mags = radii**2.0
    new_csa_mags.add_(negative_csa_deltas)

    # Grow muscle from cyt, if possible
    cyt_desired = torch.sum(positive_csa_deltas, dim=1)
    cyt_desired_safe = torch.where(cyt_desired == 0, torch.tensor(1.0), cyt_desired)
    csa_delta_distribution = positive_csa_deltas / cyt_desired_safe.unsqueeze(1)

    cyt_consumed = torch.where(cyt_desired > cytoplasm, cytoplasm, cyt_desired)
    cytoplasm.sub_(cyt_consumed)
    cyt_consumed.mul_(efficiency)
    new_csa_mags.addcmul_(cyt_consumed.unsqueeze(1), csa_delta_distribution)
    torch.mul(torch.sqrt(new_csa_mags), torch.sign(radii + radii_deltas), out=radii)  

    return radii, cytoplasm


def grow_old(rads_b, rad_deltas_b, cyt_b, efficiency_b):
    csa_deltas_b = (rads_b + rad_deltas_b)**2 - rads_b**2 # cross-sectional area

    # Atrophy muscle and convert to cyt
    cyt_b -= torch.sum(torch.where(csa_deltas_b < 0, csa_deltas_b, torch.tensor(0.0)),dim=1) * efficiency_b

    new_csa_mags_b = rads_b**2.0
    new_csa_mags_b[csa_deltas_b < 0] += csa_deltas_b[csa_deltas_b < 0]

    # Grow myscle from cyt, if possible
    cyt_desired_b = torch.sum(torch.where(csa_deltas_b > 0, csa_deltas_b, torch.tensor(0.0)),dim=1)
    csa_delta_distribution_b = torch.where(csa_deltas_b > 0, csa_deltas_b, torch.tensor(0.0)) / cyt_desired_b.unsqueeze(1)

    cyt_consumed_b = torch.where(cyt_desired_b > cyt_b, cyt_b, cyt_desired_b)
    csa_grown_b = cyt_consumed_b * efficiency_b
    new_csa_mags_b = torch.where(csa_deltas_b > 0, new_csa_mags_b + csa_grown_b.unsqueeze(1) * csa_delta_distribution_b, new_csa_mags_b)

    cyt_b -= cyt_consumed_b

    new_rad_mags_b = torch.sqrt(new_csa_mags_b)
    new_signs = torch.sign(rads_b + rad_deltas_b)

    return new_rad_mags_b * new_signs, cyt_b


def grow_dumb(rads, rad_deltas, cyt, efficiency):
    activate_muscle_growth = np.vectorize(lambda rads, rad_deltas: (rads + rad_deltas)**2 - rads**2)
    csa_deltas = activate_muscle_growth(rads, rad_deltas) # cross-sectional area
    positive_csa_deltas = csa_deltas[csa_deltas > 0]
    negative_csa_deltas = csa_deltas[csa_deltas < 0]

    # Atrophy muscle and convert to cyt
    cyt -= sum(negative_csa_deltas) * efficiency
    new_csa_mags = rads**2.0
    new_csa_mags[csa_deltas < 0] += negative_csa_deltas

    # Grow muscle from cyt, if possible
    cyt_desired = sum(positive_csa_deltas) # before efficiency loss
    csa_delta_distribution = positive_csa_deltas / cyt_desired

    if cyt_desired > cyt:
        cyt_consumed = cyt
    else:
        cyt_consumed = cyt_desired

    csa_grown = cyt_consumed * efficiency
    new_csa_mags[csa_deltas > 0] += csa_grown * csa_delta_distribution

    cyt -= cyt_consumed

    new_rad_mags = np.sqrt(new_csa_mags)
    new_signs = np.sign(rads + rad_deltas)


if __name__ == "__main__":
    # Tests if cytoplasm is consumed correctly and radii are updated in place

    import matplotlib.pyplot as plt

    rads_batch = torch.tensor([[-4]], dtype=torch.float32)
    rad_deltas_batch = torch.tensor([[2]], dtype=torch.float32)
    cytoplasm_batch = torch.tensor([3.0], dtype=torch.float32)
    efficiency_batch = torch.tensor([1.0], dtype=torch.float32)

    radii_results = []
    cytoplasm_results = []
    for _ in range(10):
        out = grow_muscle(rads_batch, rad_deltas_batch, cytoplasm_batch, efficiency_batch)
        radii_results.append(out[0].item())
        cytoplasm_results.append(out[1].item())

    assert torch.isclose(rads_batch, torch.tensor([[4.358899]], dtype=torch.float32), atol=1e-6).all()
    assert torch.isclose(cytoplasm_batch, torch.tensor([0.0], dtype=torch.float32), atol=1e-6).all()

    plt.figure(figsize=(10, 5))
    plt.plot(radii_results, label='Radii')
    plt.plot(cytoplasm_results, label='Cytoplasm')
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title('Growth of Muscle Tissue')
    plt.legend()
    plt.grid(True)
    plt.show()