import taichi as ti
import torch
from src_ti.eincasm import eincasm
from src_ti.visualize import PixelVis

ti.init(ti.gpu)
ein = eincasm(shape=(50,50), torch_device=torch.device('mps'))

def update_world():
    return 0

vis = PixelVis(ein.world, {
    'capital': 'viridis',
    'waste': 'hot',
    'port': 'copper',
    'obstacle': 'gray'
}, update_world)

vis.launch()
