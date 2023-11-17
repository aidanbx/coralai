import taichi as ti
from taichi.math import cmul, dot, log2, vec2, vec3

ti.init(arch=ti.gpu)

width, height = 800, 640
pixels = ti.Vector.field(3, ti.f32, shape=(width, height))


@ti.func
def setcolor(z, i):
    v = log2(i + 1 - log2(log2(z.norm()))) / 5
    col = vec3(0.0)
    if v < 1.0:
        col = vec3(v**4, v**2.5, v)
    else:
        v = ti.max(0.0, 2 - v)
        col = vec3(v, v**1.5, v**3)
    return col


@ti.kernel
def render(iteration: ti.f16):
    start_zoom = 0.64
    end_zoom = 0.35
    zoom = ti.cos(0.006 * iteration)
    maxiters = 50 + 10 * ti.sin(0.006 * iteration)
    zoo = start_zoom + end_zoom * zoom
    zoo = ti.pow(zoo, 8.0)
    # ca = ti.cos(0.15 * (1.0 - zoo) * time)
    # sa = ti.sin(0.15 * (1.0 - zoo) * time)
    for i, j in pixels:
        c = 2.0 * vec2(i, j) / height - vec2(1)
        c *= 0.9
        xy = vec2(c.x - c.y, c.x + c.y)
        c = vec2(-0.745, 0.186) + xy * zoo
        z = vec2(0.0)
        count = 0.0
        while count < maxiters and dot(z, z) < 50:
            z = cmul(z, z) + c
            count += 1.0

        if count == maxiters:
            pixels[i, j] = [0, 0, 0]
        else:
            pixels[i, j] = setcolor(z, count)


def main():
    gui = ti.GUI("Mandelbrot set zoom", res=(width, height))

    for iteration in range(100000):
        render(iteration)
        gui.set_image(pixels)
        gui.show()


if __name__ == "__main__":
    main()
