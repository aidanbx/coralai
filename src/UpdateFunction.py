from src import utils
from src.Channel import Channel

class UpdateFunction:
    def __init__(self, id: str, func: callable,
                 input_channel_ids: list[str] = [], affected_channel_ids: list[str] = [],
                 metadata: dict = None, req_sim_metadata: dict = {},
                 req_channel_metadata: dict = {}):
        self.id = id
        self.func = func
        self.input_channel_ids = input_channel_ids
        self.affected_channel_ids = affected_channel_ids
        self.req_sim_metadata = req_sim_metadata
        self.req_channel_metadata = req_channel_metadata
        
        default_metadata = {
            'id': id,
            'input_channel_ids': input_channel_ids,
            'affected_channel_ids': affected_channel_ids,
            'req_sim_metadata': req_sim_metadata,
            'req_channel_metadata': req_channel_metadata,
        }
        if metadata is None:
            metadata = {}
        metadata.update(default_metadata)
        self.metadata = metadata

    def assert_compatability(self, sim):
        for id in self.input_channel_ids:
            assert id in sim.channels.keys(), f"Input channel \"{id}\" for update func \"{self.id}\" not in simulation \"{sim.id}\""
        for id in self.affected_channel_ids:
            assert id in sim.channels.keys(), f"Affected channel \"{id}\" for update func \"{self.id}\" not in simulation \"{sim.id}\""
        
        utils.check_subdict(sim.metadata, self.req_sim_metadata)
        all_channel_metadata = {}
        for channel_id, channel in sim.channels.items():
            all_channel_metadata[channel_id] = channel.metadata
        utils.check_subdict(all_channel_metadata, self.req_channel_metadata)


    def update(self, sim):
        try:
            self.func(sim, *[sim.channels[id] for id in self.input_channel_ids], self.metadata)
        except Exception as e:
            raise RuntimeError(f"UpdateFunction \"{self.id}\": Error in function execution: {str(e)}")
        for ch in self.affected_channel_ids:
            if ch.shape != ch.contents.shape:
                raise RuntimeError(f"UpdateFunction \"{self.id}\": Affected Channel \"{ch.id}\" has shape {ch.shape}, but updated contents have shape {ch.contents.shape}")

# class Agent:
#     def __init__(self, config_file):
#         self.cfg = EINCASMConfig.Config(config_file)
#         self.init_io()

#     def random_agent(self):
#         actuators = torch.rand(self.total_actuator_channels)*2-1
#         return actuators

#     def grow(self, actuators, muscle_radii, capital, growth_cost):
#         muscle_radii_delta = actuators[self.actuator_indecies["muscle_radii_delta"]]
#         physics.grow_muscle_csa(self.cfg, muscle_radii, muscle_radii_delta, capital, growth_cost)

#     def flow(self, actuators, capital, waste, muscle_radii, flow_cost, obstacles):
#         flow_act = actuators[self.actuator_indecies["flow_act"]]
#         physics.activate_flow_muscles(self.cfg, capital, waste, muscle_radii[:-2], flow_act, flow_cost, obstacles)

#     def eat(self, actuators, capital, port, muscle_radii, port_cost):
#         port_act = actuators[self.actuator_indecies["port_act"]]
#         physics.activate_port_muscles(self.cfg, capital, port, muscle_radii[-2], port_act, port_cost)

#     def dig(self, actuators, muscle_radii, mine_act, capital, obstacles, waste, mining_cost):
#         mine_act = actuators[self.actuator_indecies["mine_act"]]
#         physics.activate_mine_muscles(muscle_radii[-1], mine_act, capital, obstacles, waste, mining_cost)


#     def init_io(self):
#         num_scents = self.cfg.num_resource_types
#         num_com = self.cfg.num_communication_channels * self.cfg.kernel.shape[0]
#         class Sensors:
#             def __init__(self, kernel_size, num_dims, num_scents, num_com):
#                 start_index = 0
#                 self.all_sensors = np.array([])

#                 self.scent = np.arange(start_index, num_scents)
#                 np.append(self.all_sensors, self.scent)
#                 start_index += num_scents

#                 self.capital = np.arange(start_index, kernel_size)
#                 np.append(self.all_sensors, self.capital)
#                 start_index += kernel_size

#                 self.muscles = np.arange(start_index, kernel_size)
#                 np.append(self.all_sensors, self.muscles)
            
#             def add_sensor(self, size):
#                 np.append(self.all_sensors

#             def gen_kernel_perception(self, kernel_size):
#                 return 

#         class actuators:
#             self.actuator

#         start_index = 0
#         self.perception_indecies = {
#             "scent": ,
#         }
#         start_index += num_scents

#         # Single channel kernel perception
#         for key in ["capital", "muscles"]:
#             self.perception_indecies[key] = generate_indices(start_index, self.cfg.kernel.shape[0])
#             start_index += self.cfg.kernel.shape[0]

#         self.perception_indecies["communication"] = generate_indices(start_index, num_com)
#         self.total_perception_channels = start_index + num_com

#         start_index = 0
#         self.actuator_indecies = {
#             "flow_act":  start_index,
#             "mine_act":  start_index + 1,
#             "port_act":  start_index + 2,
#         }
#         start_index += 3

#         self.actuator_indecies["muscle_radii_delta"] = generate_indices(start_index, self.cfg.num_muscles)
#         start_index += self.cfg.num_muscles

#         self.actuator_indecies["communication"] = generate_indices(start_index, self.cfg.num_communication_channels)
#         self.total_actuator_channels = start_index + self.cfg.num_communication_channels