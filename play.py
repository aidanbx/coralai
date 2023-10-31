import taichi as ti

ti.init(ti.gpu)

cmaps = [
    ti.field(dtype=ti.f32, shape=(100,100)),
    ti.field(dtype=ti.f32, shape=(10,10)),
]

@ti.kernel
def f():
    for i in ti.static(range(2)):
        cmaps[i][0,0] = cmaps[i].shape[0]

f()

print(cmaps[0][0,0], cmaps[1][0,0])