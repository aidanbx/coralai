import numpy as np
import taichi as ti
import torch
from src_ti.eincasm import eincasm
from src_ti.visualize import PixelVis

ti.init(ti.gpu)

ein = eincasm(shape=(50,50), torch_device=torch.device('mps'))
ein.world['capital'] = torch.ones_like(ein.world['capital'])
ein.world.indices.return_ti(True)

@ti.func
def tfunc(mem: ti.types.ndarray(), cid: ti.template(), mids: ti.template()):
    for i,j in ti.ndrange(mem.shape[0], mem.shape[1]):
        for mid in range(mids.shape[0]):
            mem[i, j, mids[mid]] += mem[i, j, cid[0]]

@ti.kernel
def test(mem: ti.types.ndarray()):
    tfunc(mem, ein.world.indices['capital'], ein.world.indices['muscles'])
test(ein.world.mem)
print(ein.world['muscles'])

exit()
ein = eincasm(shape=(50,50), torch_device=torch.device('mps'))

def update_world():
    ein.apply_ti_physics(ein.world.mem)

vis = PixelVis(ein.world, {
    'capital': 'viridis',
    'waste': 'hot',
    'port': 'copper',
    'obstacle': 'gray'
}, update_world)

print(ein.world['capital'].shape)
print(vis.ch_lims)
vis.launch()
print(ein.world['obstacle'].min(), ein.world['obstacle'].max())