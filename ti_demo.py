import taichi as ti
import src_ti.eincasm as eincasm
import src_ti.pcg as pcg_ti

ti.init(arch=ti.gpu)

SIM_WIDTH = 640
SIM_HEIGHT = 480

# einfield = eincasm.eincell.field(shape=(sim_width, sim_height))

gray_scale_image = ti.field(dtype=ti.f32, shape=(SIM_WIDTH, SIM_HEIGHT))
einfield = eincasm.Cell.field(shape=(SIM_WIDTH, SIM_HEIGHT))
einfield.bla = 2

@ti.func
def init_sim(state):
    for i,j in state:
        state[i,j]['ob'] = ti.random()

@ti.kernel
def fill_image(state: ti.template()):
    init_sim(state)
    # Fills the image with random gray
    for i,j in gray_scale_image:
        gray_scale_image[i,j] = state[i,j].ob

# fill_image(einfield)
# print(einfield.keys())
# # Creates a GUI of the size of the gray-scale image
# gui = ti.GUI('gray-scale image of random values', (SIM_WIDTH, SIM_HEIGHT))
# while gui.running:
#     gui.set_image(gray_scale_image)
#     gui.show()


