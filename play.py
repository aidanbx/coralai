import numpy as np
import taichi as ti
import torch
from src_ti.eincasm import eincasm
from src_ti.PixelVis import PixelVis
from src_ti.Vis import Vis
import src_ti.physics as physics
from src_ti.TaichiStructFactory import TaichiStructFactory

ti.init(ti.gpu)

builder = TaichiStructFactory()
builder.add_ti_i('test', 2)
vars = builder.build()

@ti.kernel
def test(vars: ti.template()):
    v=vars[None]
    v.test = 123123123
    vars[None] = v

test(vars)
print(vars[None].test)