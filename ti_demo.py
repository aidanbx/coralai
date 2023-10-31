import taichi as ti
import torch
from src_ti.eincasm import eincasm
from src_ti.visualize import PixelVis

ti.init(ti.gpu)
ein = eincasm(shape=(3,3), torch_device=torch.device('mps'))
ein.world.malloc()
print(ein.world.index[['capital', 'waste', 'muscles', 'obstacle']])

# ein.world.malloc()

# vis = PixelVis(ein.world, ['capital', 'waste', 'port', 'obstacle'])

# while True:
#     # ein.apply_rules()
#     vis.update()
#     # vis.render()