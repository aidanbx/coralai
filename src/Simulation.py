import numpy as np
import torch
import importlib
import src.Channel as Channel
importlib.reload(Channel)

class Simulation:
    def __init__(self):
        self.channels = []
        self.channel_ids = []
        self.update_functions = []
        self.channel_data = {}

    def add_channel(self, channel: Channel.Channel):
        self.channels.append(channel)
        self.channel_ids.append(channel.id)

    def add_channel(self, id, shape, init_func, metadata=None,
                    allowed_range=None, dtype=torch.float32, device=torch.device("cpu")):
        
        self.add_channel(Channel.Channel(id, shape, init_func, metadata,
                                         allowed_range, dtype, device))

    def add_update_function(self, update_function: UpdateFunction):
        for id in update_function.input_channel_ids:
            assert id in self.channel_ids, f"Channel {id} not in simulation"
        for id in update_function.affected_channels:
            assert id in self.channel_ids, f"Channel {id} not in simulation"
        self.update_functions.append(function)
    
    def init_all_channels(self):
        for channel in self.channels:
            self.channel_data[channel.id] = channel.init()



