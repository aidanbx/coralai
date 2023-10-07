import importlib
import src.Simulation as Simulation
importlib.reload(Simulation)
import src.Channel as Channel
importlib.reload(Channel)
import src.utils as utils
importlib.reload(utils)

class UpdateFunction:
    def __init__(self, id: str, func: callable,
                 input_channel_ids: list[str], affected_channel_ids: list[str],
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

    def assert_compatability(self, sim: Simulation.Simulation):
        for id in self.input_channel_ids:
            assert id in sim.channels.keys(), f"Input channel \"{id}\" for update func \"{self.id}\" not in simulation \"{sim.id}\""
        for id in self.affected_channel_ids:
            assert id in sim.channels.keys(), f"Affected channel \"{id}\" for update func \"{self.id}\" not in simulation \"{sim.id}\""
        
        utils.check_subdict(sim.metadata, self.req_sim_metadata)
        all_channel_metadata = {}
        for channel_id, channel in sim.channels.items():
            all_channel_metadata[channel_id] = channel.metadata
        utils.check_subdict(all_channel_metadata, self.req_channel_metadata)

    def update(self, sim: Simulation.Simulation):
        # assert self.contents is not None, f"Channel \"{self.id}\" has not been initialized"
        output = self.func(sim, *[sim.channels[id] for id in self.input_channel_ids])
        
        desired_output_size = len(self.affected_channel_ids)
        output_size = 0
        output_size = 1 if isinstance(output, Channel.Channel) else output_size
        output_size = len(output) if isinstance(output, list) else output_size 
        assert output_size == desired_output_size, (f"Update function \"{self.id}\" must return {desired_output_size}" +
                                                    f"Channel(s) in order: {self.affected_channel_ids}, got {output_size}" +
                                                    f"Channel(s): {output} instead")