import numpy as np
import taichi as ti
import torch
from src_ti.eincasm import eincasm
from src_ti.PixelVis import PixelVis
from src_ti.Vis import Vis
import src_ti.physics as physics
from src_ti.TaichiStructFactory import TaichiStructFactory


ti.init(ti.gpu)
mystruct = ti.types.struct(**{
    "f1": ti.types.vector(n=2, dtype=ti.f32),
    "i1": ti.types.vector(n=2, dtype=ti.i32),
})

myfield = mystruct.field(shape=())
myfield[None]['f1'] = [1.0, 2.0]
myfield[None]['i1'] = [1, 2]