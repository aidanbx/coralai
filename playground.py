# %% Physics ------------------------------------------------------------------
import numpy as np
max_delta_muscle = 5 # absolute value
growth_efficiency = 0.8

# growth_cost = np.vectorize(lambda old, delta: ((old+delta)**2 - abs(old))/growth_efficiency)
activate_muscle_growth = np.vectorize(lambda old, delta: ((old+delta)**2 - old**2))
muscle_strength = np.vectorize(lambda muscle: muscle**2)

# %% Env and prev values ------------------------------------------------------
kernel = np.array([[0,0],[0,-1],[1,0],[0,-1],[-1,0]])
obstacle = np.array([0,0,0,1,0])
capital = np.array([2,4,5,0.2,0.2])

muscle_radii = np.array([0.2,2,-3,0.1,0.1])   # single cell's muscles
flows = np.array([0,0,0,0,0])           # brand new cell
signals = np.array([0,0,0,0,0])         # brand new cell

# %% Physiological outputs -----------------------------------------------------
delta_muscle_radius = np.array([0.1,-0.2,1,3,0.1])
activation = -0.8
signals = np.array([0,0,0,0,0]) # doesn't matter for physics

# %% Apply Physics -------------------------------------------------------------
delta_muscle_radius = np.clip(delta_muscle_radius, -max_delta_muscle, max_delta_muscle)
delta_muscle_mass = activate_muscle_growth(muscle_radii, delta_muscle_radius)
heat_loss = np.abs(delta_muscle_mass) * (1.0-growth_efficiency)
capital_needed = sum(delta_muscle_mass + heat_loss)

# %%
import torch

def activate_muscle_growth(rads, rad_deltas):
    return (rads + rad_deltas)**2 - rads**2

def grow_muscle(rads_batch, rad_deltas_batch, capital_batch, efficiency_batch):
    csa_deltas = (rads_batch + rad_deltas_batch)**2 - rads_batch**2 # cross-sectional area
    positive_csa_deltas = csa_deltas[csa_deltas > 0]
    negative_csa_deltas = csa_deltas[csa_deltas < 0]

    # Atrophy muscle and convert to capital
    capital_batch = capital_batch + torch.sum(rads_batch, dim=1) * efficiency_batch
    new_csa_mags = rads_batch**2.0
    new_csa_mags[csa_deltas < 0] = new_csa_mags[csa_deltas < 0] + negative_csa_deltas

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

    return new_rad_mags * new_signs, capital

# Define your batches
rads_batch = torch.tensor([[0.2, 2, -3, 0.1, 0.1], [0.3, 2.5, -2, 0.2, 0.2], [0.4, 3, -1, 0.3, 0.3]])
rad_deltas_batch = torch.tensor([[0.1, -0.2, 1, 3, 0.1], [0.2, -0.3, 2, 2, 0.2], [0.3, -0.4, 3, 1, 0.3]])
capital_batch = torch.tensor([2, 3, 4], dtype=torch.float32)
efficiency_batch = torch.tensor([0.8, 0.85, 0.9], dtype=torch.float32)

# Call the function with batched inputs
new_rads_batch, new_capital_batch = grow_muscle(rads_batch, rad_deltas_batch, capital_batch, efficiency_batch)









