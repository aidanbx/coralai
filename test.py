import torch
import matplotlib.pyplot as plt

# Sample from the distribution
tensor = (torch.randn(10000)-1) * 0.1

flat_np = tensor.flatten().numpy()
plt.hist(tensor.numpy(), bins=100)
plt.show()
