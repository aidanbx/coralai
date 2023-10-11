import torch
import numpy as np
from src.UpdateFunction import UpdateFunction
from src.Channel import Channel

class Simulation:
    def __init__(self, id: str = "default", world_shape: tuple = (100, 100), metadata: dict = None):
        self.id = id
        self.world_shape = world_shape
        self.channels = {}
        self.update_functions = {}
        self._deferred_groups = {}

        default_metadata = {
            'id': id,
            'world_shape': world_shape,
        }
        self.metadata = {**metadata, **default_metadata} if metadata else default_metadata

    def add_channel(self, id: str, init_func: callable = None, num_layers: int = 1, metadata: dict = None,
                    allowed_range: tuple = None, dtype=torch.float32, device=torch.device("cpu")):
        if id in self.channels:
            raise ValueError(f"Channel with ID {id} already exists.")
        self.channels[id] = Channel(id, self.world_shape, num_layers, init_func, metadata,
                                    allowed_range, dtype, device)

    def add_update_function(self, id, function, input_channel_ids, affected_channel_ids,
                            metadata: dict = None, req_sim_metadata: dict = {},
                            req_channel_metadata: dict = {}):
        if id in self.update_functions:
            raise ValueError(f"UpdateFunction with ID {id} already exists.")
        update_func = UpdateFunction(id, function, input_channel_ids, affected_channel_ids,
                                     metadata, req_sim_metadata, req_channel_metadata)
        update_func.assert_compatability(sim=self)
        self.update_functions[id] = update_func

    def update_group(self, old_group_id, new_group_id):

    def define_channel_group(self, group_id, channel_ids: np.array, group_metadata, dtype=torch.float32, device=torch.device("cpu")):
        if group_id in self.channels:
            raise ValueError(f"Channel with ID {group_id} already exists.")

        flat_chids = set(channel_ids)
        subgroups = []
        chs_in_groups = set()
        for ch_id in channel_ids:
            if ch_id not in self.channels:
                raise ValueError(f"Channel with ID {ch_id} does not exist.")
            channel = self.channels[ch_id]
            if channel.metadata['is_group']:
                subgroups.append(ch_id)
                for ch_ 
                [flat_chids.add(grouped_ch_id) for grouped_ch_id in channel.metadata['channel_ids']]

        num_layers = sum([self.channels[ch_id].num_layers for ch_id in flat_chids])

        # Add group metadata
        additional_metadata = {
            'is_group': True,
            'num_layers': num_layers,     # for convenience
            'channel_ids': channel_ids,
            'flat_chids': flat_chids,
            'subgroup_ids': subgroups,
        }
        final_metadata = {**group_metadata, **additional_metadata}

        # Add the group tensor as a new Channel with appropriate metadata
        def init_group(shape, metadata):

            tensors = []
            for ch_id in metadata['flat_chids']:
                channel = self.channels[ch_id]
                if not channel.initialized:
                    channel.init_contents()
                tensors.append(channel.contents)
            
            group_tensor = torch.cat(tensors, dim=0)

            # redefine the base tensors in terms of this highest level (as of now) group
            start_index = 0
            for ch_id in metadata['flat_chids']:
                channel = self.channels[ch_id]
                del channel.contents
                num_layers = channel.num_layers
                channel.contents = group_tensor[start_index:start_index+num_layers].view(channel.num_layers, *self.world_shape)

            for gr_id in metadata['subgroups']:
                group_ch = self.channels[gr_id]
                del group_ch.contents
                group_ch.

            return group_tensor, {}
                

        self.add_channel(group_id, init_group, group_shape[0], metadata=final_metadata, dtype=dtype, device=device)

    def init_all_channels(self):
        for channel_id, channel in self.channels.items():
            if not channel.initialized:
                channel.init_contents()
    
    def update(self):
        for _, update_function in self.update_functions.items():
            update_function.update(sim=self)
