import torch
import taichi as ti

ti.init(ti.gpu)
device = torch.device('mps:0')

depth_distr1 = torch.randn((128,128,32), device=device)
depth_distr2 = depth_distr1.clone()

from timeit import default_timer as timer

print(depth_distr1.sum())
print(depth_distr2.sum())

start = timer()
distribute(depth_distr1)
print(depth_distr1.sum())
end = timer()
print(end-start)

start = timer()
sum_half = torch.sum(depth_distr2, dim=2)
depth_distr2.div_(sum_half.unsqueeze(2))
print(depth_distr2.sum())
end = timer()
print(end-start)