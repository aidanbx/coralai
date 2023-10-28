import taichi as ti
from src_ti.eincasm import eincasm
# from src_ti.simvis import simvis

ti.init(ti.gpu)
ein = eincasm(shape=(3,3))
ein.world.malloc()
print(ein.world.mem)