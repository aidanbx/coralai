import os
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from timeit import default_timer as timer
import importlib
import pcg as gen_env
importlib.reload(gen_env)
import src.physics as physics
importlib.reload(physics)
import src.EINCASMConfig as EINCASMConfig
importlib.reload(EINCASMConfig)
import src.Resource as Resource
importlib.reload(Resource)

class Agent:
    def __init__(self, config_file):
        self.cfg = EINCASMConfig.Config(config_file)
        self.init_io()

    def random_agent(self):
        actuators = torch.rand(self.total_actuator_channels)*2-1
        return actuators

    def grow(self, actuators, muscle_radii, capital, growth_cost):
        muscle_radii_delta = actuators[self.actuator_indecies["muscle_radii_delta"]]
        physics.grow_muscle_csa(self.cfg, muscle_radii, muscle_radii_delta, capital, growth_cost)

    def flow(self, actuators, capital, waste, muscle_radii, flow_cost, obstacles):
        flow_act = actuators[self.actuator_indecies["flow_act"]]
        physics.activate_flow_muscles(self.cfg, capital, waste, muscle_radii[:-2], flow_act, flow_cost, obstacles)

    def eat(self, actuators, capital, port, muscle_radii, port_cost):
        port_act = actuators[self.actuator_indecies["port_act"]]
        physics.activate_port_muscles(self.cfg, capital, port, muscle_radii[-2], port_act, port_cost)

    def dig(self, actuators, muscle_radii, mine_act, capital, obstacles, waste, mining_cost):
        mine_act = actuators[self.actuator_indecies["mine_act"]]
        physics.activate_mine_muscles(muscle_radii[-1], mine_act, capital, obstacles, waste, mining_cost)


    def init_io(self):
        num_scents = self.cfg.num_resource_types
        num_com = self.cfg.num_communication_channels * self.cfg.kernel.shape[0]
        class Sensors:
            def __init__(self, kernel_size, num_dims, num_scents, num_com):
                start_index = 0
                self.all_sensors = np.array([])

                self.scent = np.arange(start_index, num_scents)
                np.append(self.all_sensors, self.scent)
                start_index += num_scents

                self.capital = np.arange(start_index, kernel_size)
                np.append(self.all_sensors, self.capital)
                start_index += kernel_size

                self.muscles = np.arange(start_index, kernel_size)
                np.append(self.all_sensors, self.muscles)
            
            def add_sensor(self, size):
                np.append(self.all_sensors

            def gen_kernel_perception(self, kernel_size):
                return 

        class actuators:
            self.actuator

        start_index = 0
        self.perception_indecies = {
            "scent": ,
        }
        start_index += num_scents

        # Single channel kernel perception
        for key in ["capital", "muscles"]:
            self.perception_indecies[key] = generate_indices(start_index, self.cfg.kernel.shape[0])
            start_index += self.cfg.kernel.shape[0]

        self.perception_indecies["communication"] = generate_indices(start_index, num_com)
        self.total_perception_channels = start_index + num_com

        start_index = 0
        self.actuator_indecies = {
            "flow_act":  start_index,
            "mine_act":  start_index + 1,
            "port_act":  start_index + 2,
        }
        start_index += 3

        self.actuator_indecies["muscle_radii_delta"] = generate_indices(start_index, self.cfg.num_muscles)
        start_index += self.cfg.num_muscles

        self.actuator_indecies["communication"] = generate_indices(start_index, self.cfg.num_communication_channels)
        self.total_actuator_channels = start_index + self.cfg.num_communication_channels