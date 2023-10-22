import taichi as ti
import torch
import torch.nn as nn

ti.init()

SIM_WIDTH = 640
SIM_HEIGHT = 480

COM_CHANNELS = 5

flow_kernel = ti.Vector.field(2, dtype=ti.i8, shape=5)
flow_kernel[0] = [0, 0]  # ORIGIN
flow_kernel[1] = [-1, 0] # UP
flow_kernel[2] = [0, 1]  # RIGHT
flow_kernel[3] = [1, 0]  # DOWN
flow_kernel[4] = [0, -1] # LEFT

@ti.dataclass
class Cell:
    ob: ti.f32
    ports: ti.f32
    capital: ti.f32
    waste: ti.f32
    comm: ti.types.vector(COM_CHANNELS, dtype=ti.f32)
    flow_m: ti.types.vector(flow_kernel.shape[0], dtype=ti.f32)
    port_m: ti.f32
    mine_m: ti.f32
    growth_act: ti.types.vector(flow_kernel.shape[0] + 2, dtype=ti.f32)
    flow_act: ti.f32



