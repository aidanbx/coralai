import taichi as ti
import src.pcg as pcg_base

@ti.func
def init_ob(einfield):
    for i,j in einfield:
        einfield[i,j].ob = ti.random()