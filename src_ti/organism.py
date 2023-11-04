# import numpy as np
# import taichi as ti
# import torch
# from src_ti.world import World

# @ti.data_oriented
# class organism:
#     def __init__(self, world: World):
#         self.world = world
#         sense_kernel = np.array([
#                 [0, 0],  # ORIGIN
#                 [-1, 0],  # UP
#                 [0, 1],   # RIGHT
#                 [1, 0],   # DOWN
#                 [0, -1], # LEFT
#         ])
#         sense_kernel = ti.Vector.field(2, dtype=ti.i32, shape=5)
#         self.sense_kernel = sense_kernel

#         self.senses_neigh = {
#             'capital': False, # senses neighborhood?
#             'muscles': False,
#             'com': True,
#             'obstacle': True
#         }
#         self.sensors = self.senses_neigh.keys()
#         self.actuators = ['muscle_acts', 'growth_acts', 'com']

#     def gen_sensor_weights(self):
#         for sensor in self.sensors:
#         all_sensor_weights = []
#         for genome_id in range(len(sensor_sizes)): # num_genomes
#             w = {}
#             for sensor, senses_neigh in sensors.items():
#                 if senses_neigh:
#                     w[sensor] = ti.field(ch_types[sensor], shape=(sense_kernel.shape[0], sensor_sizes[genome_id][sensor]))
#                 else:
#                     w[sensor] = ti.field(ch_types[sensor], shape=(sensor_sizes[genome_id][sensor]))
#             all_sensor_weights.append(w)
#         return all_sensor_weights

# sensor_weights = gen_sensor_weights(sensors, ch_types, sense_kernel, sensor_sizes)

# sensor_layers = []
# for genome_ss in sensor_sizes:
#     layers = {}
#     for sensor, layer_size in genome_ss.items():
#         layers[sensor] = ti.types.vector(n=layer_size, dtype=ti.f32)
#     sensor_layers.append(ti.types.struct(**layers))

# print(f"Sensor Layers: {sensor_layers}\n\nSensor Weights: {sensor_weights}")

# # @ti.kernel
# # def forward():
    
        
# # def gen_actuator_weights(actuators, ch_types, actuator_sizes):
# #     all_actuator_weights = []
# #     for genome_id in range(len(actuator_sizes)): # num genomes
# #         w = {}
# #         for actuator in actuators:
# #             w[actuator] = ti.field(ch_types[actuator], shape=(latent_sizes[genome_id], actuator_sizes[genome_id][actuator]))
# #         all_actuator_weights.append(w)
# #     return all_actuator_weights

# # def gen_latent_weights(sensor_sizes, latent_sizes, actuator_sizes):
# #     incoming_latent_weights = []
# #     outgoing_latent_weights = []
# #     for genome_id in range(num_genomes):
# #         sensor_latent_w = {}
# #         for sensor in sensors.keys():
# #             sensor_latent_w[sensor] = ti.field(ti.f32, shape=(sensor_sizes[genome_id][sensor], latent_sizes[genome_id]))

# #         latent_actuator_w = {}
