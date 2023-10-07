import torch
from pyperlin import FractalPerlin2D
import importlib
import random
import src.EINCASMConfig as EINCASMConfig
importlib.reload(EINCASMConfig)
import src.Simulation as Simulation
importlib.reload(Simulation)
import src.pcg as pcg
importlib.reload(pcg)
import src.Resource as Resource
importlib.reload(Resource)
import src.Channel as Channel
importlib.reload(Channel)
import src.physics as physics
importlib.reload(physics)
import src.utils as utils
importlib.reload(utils)
import src.update_funcs as update_funcs
importlib.reload(update_funcs)
import src.channel_funcs as channel_funcs
importlib.reload(channel_funcs)

class EINCASM:
    def __init__(self, config_file):
        self.cfg = EINCASMConfig.Config(config_file)
        self.sim = Simulation.Simulation("EINCASM Experiment")
        self.define_channels()

    def define_channels(self):
        self.sim.add_channel("obstacles", init_func=channel_funcs.init_obstacles_perlin,
                             metadata={'description': 'Continuous obstacles with perlin noise'})
        
        self.sim.add_channel("ports", init_func=channel_funcs.init_resources_levy, allowed_range=[-1,10],
                             metadata={'description': 'Currently +/- resources generated with levy dust',
                                       'num_resources': 3,
                                       'min_regen_amp': 0.5,
                                       'max_regen_amp': 2,
                                       'alpha_range': [0.4, 0.9],
                                       'beta_range': [0.8, 1.2],
                                       'num_sites_range': [50, 100]})

        self.sim.metadata.update({'period': 0.0})
        self.sim.add_update_function("step_period",
                                     update_funcs.step_period,
                                     input_channel_ids=[], affected_channel_ids=[],
                                     metadata={'description': 'Increment period'},
                                     req_sim_metadata = {"period": float})

        self.sim.add_update_function("regen_resources",
                                     update_funcs.regen_ports,
                                     input_channel_ids=["ports", "obstacles"], affected_channel_ids=["ports"],
                                     metadata={'description': 'Regenerate resources'},
                                     req_channel_metadata = {"ports": ["port_id_map", "port_sizes", "resources"]},
                                     req_sim_metadata = {"period": float})