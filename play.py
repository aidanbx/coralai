import taichi as ti

ti.init(arch=ti.gpu)

flow_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
flow_kernel[0] = [0, 0]  # ORIGIN
flow_kernel[1] = [-1, 0] # UP
flow_kernel[2] = [0, 1]  # RIGHT
flow_kernel[3] = [1, 0]  # DOWN
flow_kernel[4] = [0, -1] # LEFT

sense_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
sense_kernel[0] = [0, 0]  # ORIGIN
sense_kernel[1] = [-1, 0] # UP
sense_kernel[2] = [0, 1]  # RIGHT
sense_kernel[3] = [1, 0]  # DOWN
sense_kernel[4] = [0, -1] # LEFT

num_com = 3
channels = {
    'capital': ti.f32,
    'obstacle': ti.f32,
    'muscles': ti.types.vector(flow_kernel.shape[0], dtype=ti.f32),
    'com': ti.types.vector(num_com, dtype=ti.f32),
    'fact': ti.f32,
    'gracts': ti.types.vector(flow_kernel.shape[0], dtype=ti.f32),
}
celltype = ti.types.struct(**channels)
world = celltype.field(shape=(10,10))

@ti.func
def ReLu(x):
    return x if x > 0 else 0

sensors = {
    'capital': False, # senses neighborhood?
    'muscles': False,
    'com': True,
    'obstacle': True
}
ti.types.f32
actuators = ['fact', 'gracts']
num_genomes = 3
layer_lens = [
    {'capital': 1, 'muscles': 2, 'com': 3, 'obstacle': 5},
    {'capital': 6, 'muscles': 7, 'com': 1, 'obstacle': 3},
    {'capital': 1, 'muscles': 1, 'com': 6, 'obstacle': 4},
]
activation_funcs = [ti.tanh, ReLu, ti.sin]


def gen_genome_weights():
    all_sensor_weights = []
    for genome_id in range(num_genomes):
        w = {}
        for sensor, senses_neigh in sensors.items():
            if senses_neigh:
                w[sensor] = ti.field(channels[sensor], shape=(sense_kernel.shape[0], layer_lens[genome_id][sensor]))
            else:
                w[sensor] = ti.field(channels[sensor], shape=(layer_lens[genome_id][sensor]))
        all_sensor_weights.append(w)
    return all_sensor_weights

weights = gen_genome_weights()


# a=ti.types.vector(n=3, dtype=ti.f32)
# b=ti.types.vector(n=3, dtype=ti.f32)
# f=ti.f32
# ctype = ti.types.struct(a=a, b=b, f=f)
# cfield = ctype.field(shape=(3,3))

# layer_sizes = [[10,4,2]]







# @ti.kernel
# def init():
#     for i,j in cfield:
#         cfield[i,j].a = i
#         cfield[i,j].b = j
#         cfield[i,j].f = i+j

# @ti.kernel
# def tst():
#     for i,j in cfield:
#         cfield[i,j].a = aweights

# init()
# tst()
# print(cfield)
# def generate_genome_weights(num_layers, )