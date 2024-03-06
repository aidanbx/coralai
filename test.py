import torch


t = torch.tensor([[[4,8,4],[23,2,2]],[[2,3,1],[2,3,1]]], dtype=torch.float32, device="mps")
t[0,:]=1
print(t)

