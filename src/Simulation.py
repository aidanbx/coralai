import numpy as np
import torch
import importlib
import src.Channel as Channel
importlib.reload(Channel)
import src.UpdateFunc as UpdateFunction
importlib.reload(UpdateFunction)

class Simulation:
    def __init__(self, id: str = "default", world_shape: tuple = (100,100), metadata: dict = None):
        self.id = id
        self.world_shape = world_shape
        self.channels = {}
        self.update_functions = {}
        default_metadata = {
            'id': id,
            'world_shape': world_shape,
        }
        if metadata is None:
            metadata = {}
        metadata.update(default_metadata)
        self.metadata = metadata

    def _add_channel(self, channel: Channel.Channel):
        self.channels[channel.id] = channel

    def add_channel(self, id: str, init_func: callable=None, shape: tuple=None, metadata: dict=None,
                    allowed_range: tuple=None, dtype=torch.float32, device=torch.device("cpu")):
        if shape is None:
            shape = self.world_shape
        if init_func is None:
            init_func = lambda shape, metadata: torch.zeros(shape, dtype=dtype, device=device)
        self._add_channel(Channel.Channel(id, shape, init_func, metadata,
                                         allowed_range, dtype, device))

    def add_update_function(self, id, function, input_channel_ids, affected_channel_ids,
                            metadata: dict = None, req_sim_metadata: dict = {},
                            req_channel_metadata: dict = {}):
        update_func = UpdateFunction.UpdateFunction(id,function, input_channel_ids,
                                                    affected_channel_ids, metadata,
                                                    req_sim_metadata, req_channel_metadata)
        update_func.assert_compatability(sim=self)
        self.update_functions[update_func.id] = update_func

    def init_all_channels(self):
        for channel_id, channel in self.channels.items():
            channel.init_contents()
    
    def update(self):
        for update_function_id, update_function in self.update_functions.items():
            update_function.update(sim=self)
