import torch
from src import (
    Simulation,
    physics,
    pcg,
    Channel
)

# IDS
FLOW_ACTIVATION = 'flow_activation'
FLOW_MUSCLE_RADII = 'flow_muscle_radii'
PORT_MUSCLE_ACTIVATION = 'port_muscle_activation'
PORT_MUSCLE_RADII = 'port_muscle_radii'
MINE_MUSCLE_ACTIVATION = 'mine_muscle_activation'
MINE_MUSCLE_RADII = 'mine_muscle_radii'
MUSCLE_GROWTH_ACTIVATION = 'muscle_growth_activation'
MUSCLE_CH_IDS = [FLOW_MUSCLE_RADII, PORT_MUSCLE_RADII, MINE_MUSCLE_RADII]
_CH_IDS = [FLOW_ACTIVATION, PORT_MUSCLE_ACTIVATION, MINE_MUSCLE_ACTIVATION]
COMMUNICATION = 'communication'
MUSCLES = 'muscle_group'
CAPITAL = 'capital'
WASTE = 'waste'
OBSTACLES = 'obstacles'
PORTS = 'ports'

num_communication_channels = 2

class EINCASM:
    def __init__(self):
        self.sim = Simulation.Simulation('EINCASM Experiment')
        self.define_channels()
        device = 'mps'
        if device == 'mps':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                raise ValueError("MPS is not available on this system")
        elif device == 'cuda':
            self.device = torch.device("cuda")
        elif device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.float_dtype = torch.float32

        kernel = torch.tensor([
            [0, 0],     # ORIGIN
            [-1, 0],    # UP
            [0, 1.0],   # RIGHT
            [1, 0],     # DOWN
            [0, -1]     # LEFT
        ], device=self.device, dtype=torch.int8)

        assert kernel.shape[1] == self.dimensions, "Kernel must have dim(env) + 1 columns"
        assert torch.eq(kernel[0], self.zero_tensor.repeat(self.dimensions)).all(), "Kernel must have origin at index 0"
        assert kernel.shape[0] % 2 == 1, "Odd kernels (excluding origin) unimplemented"
        assert (torch.roll(kernel[1:], shifts=(kernel.shape[0]-1)//2, dims=0) - kernel[1:]).sum() == 0, "Kernel must be symmetric"
        self.kernel = kernel


    def define_channels(self):
        self.sim.add_channel(COMMUNICATION, num_layers = num_communication_channels)

        self.sim.add_channel(FLOW_ACTIVATION, allowed_range=[-1, 1])

        self.sim.add_channel(FLOW_MUSCLE_RADII,
                             num_layers = self.kernel.shape[0],
                             metadata = {'kernel': self.kernel}
                            )

        self.sim.add_channel(PORT_MUSCLE_ACTIVATION, allowed_range=[-1, 1])
        self.sim.add_channel(PORT_MUSCLE_RADII)

        self.sim.add_channel(MINE_MUSCLE_ACTIVATION, allowed_range=[-1, 1])
        self.sim.add_channel(MINE_MUSCLE_RADII)

        self.sim.define_channel_group(MUSCLES, MUSCLE_CH_IDS)

        # flow, port, and mine muscles treated equally during growth
        self.sim.add_channel(MUSCLE_GROWTH_ACTIVATION, 
                             num_layers = self.kernel.shape[0] + 2)
        

        self.sim.add_channel(CAPITAL, allowed_range = [0, 100])
        self.sim.add_channel(WASTE, allowed_range = [0, 100])
        self.sim.add_channel(OBSTACLES, init_func = pcg.init_obstacles_perlin)
        
        self.sim.add_channel(PORTS,
                             init_func = pcg.init_ports_levy, 
                             allowed_range = [-1, 10],
                             metadata = {
                                 'num_resources': 3,
                                 'min_regen_amp': 0.5,
                                 'max_regen_amp': 2,
                                 'alpha_range': [0.4, 0.9],
                                 'beta_range': [0.8, 1.2],
                                 'num_sites_range': [50, 100]
                                 }
                            )

        self.sim.metadata.update({'period': 0.0})
        self.sim.add_update_function(
            'step_period',
            lambda sim: sim.metadata.update({'period': sim.metadata['period'] + 1}),
            req_sim_metadata = {'period': float}
        )

        self.sim.add_update_function(
            'grow',
            physics.grow_muscle_csa,
            input_channel_ids = [MUSCLES, CAPITAL, MUSCLE_GROWTH_ACTIVATION],
            affected_channel_ids = [MUSCLES, CAPITAL],
            metadata = {'growth_cost': 0.2},
        )

        self.sim.add_update_function(
            'flow',
            physics.activate_flow_muscles,
            input_channel_ids = [CAPITAL, WASTE, FLOW_MUSCLE_RADII, FLOW_ACTIVATION, OBSTACLES],
            affected_channel_ids = [CAPITAL],
            metadata = {'flow_cost': 0.2, 'kernel': self.kernel},
        )

        self.sim.add_update_function(
            'eat',
            physics.activate_port_muscles,
            input_channel_ids = [CAPITAL, PORTS, OBSTACLES, PORT_MUSCLE_RADII, PORT_MUSCLE_ACTIVATION],
            affected_channel_ids = [CAPITAL],
            metadata = {'port_cost': 0.2},
        )

        self.sim.add_update_function(
            'dig',
            physics.activate_mine_muscles,
            input_channel_ids = [CAPITAL, OBSTACLES, WASTE, MINE_MUSCLE_RADII, MINE_MUSCLE_ACTIVATION],
            affected_channel_ids = [CAPITAL, WASTE],
            metadata = {'mining_cost': 0.2},
        )

        self.sim.add_update_function(
            'regen_resources',
            physics.regen_ports,
            input_channel_ids = [PORTS, OBSTACLES], 
            affected_channel_ids = [PORTS],
            req_channel_metadata = {PORTS: ['port_id_map', 'port_sizes', 'resources']},
            req_sim_metadata = {'period': float}
        )

        self.sim.add_update_function(
            'random_agent',
            physics.random_agent,
            input_channel_ids = [CAPITAL, MUSCLES, COMMUNICATION],
            affected_channel_ids = [


    def run(self):
        self.sim.init_all_channels()
        for _ in range(1000):
            self.sim.update()

