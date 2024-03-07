import torch

live_genomes = torch.tensor([1, 2, 3], device="mps")  # Example tensor for live genomes
substrate_mem = torch.tensor([[0, 1, 2, 3, 4],
                              [9, 9, 9, 9, 9],
                              [9, 9, 9, 9, 9],
                              [9, 9, 9, 9, 9]], device="mps")  # Example tensor for substrate memory

t = substrate_mem * 1
t[:] = 999
print(t)
print(substrate_mem)

