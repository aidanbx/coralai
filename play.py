import numpy as np
import taichi as ti
import torch
from eincasm.Eincasm import eincasm
from eincasm.PixelVis import PixelVis
from eincasm.Vis import Vis
import eincasm.sim.physics as physics
from eincasm.TaichiStructFactory import TaichiStructFactory

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