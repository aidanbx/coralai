import torch
import numpy as np
from src.UpdateFunction import UpdateFunction
from src.Channel import Channel

class Simulation:
    def __init__(self, id: str = "default", world_shape: tuple = (100, 100), metadata: dict = None, device: torch.device = torch.device("cpu")):
        self.id = id
        self.world_shape = world_shape
        self.world_layers = 0
        self.channels = {}
        self.update_functions = {}
        self._deferred_groups = {}
        self.device = device

        default_metadata = {
            'id': id,
            'world_shape': world_shape,
        }
        self.metadata = {**metadata, **default_metadata} if metadata else default_metadata

    def add_channel(self, id: str, init_func: callable = None, num_layers: int = 1, metadata: dict={},
                    allowed_range: tuple = None):
        if id in self.channels:
            raise ValueError(f"Channel with ID {id} already exists.")
        self.world_layers += num_layers
        shape = (num_layers, *self.world_shape)
        self.channels[id] = Channel(id, shape, init_func, metadata, allowed_range)
    
    def add_subchannel(self, id: str, parent_id: str, indices: list[int], metadata: dict={}):
        if id in self.channels:
            raise ValueError(f"Channel with ID {id} already exists, cannot add as subchannel.")
        if parent_id not in self.channels:
            raise ValueError(f"Parent channel with ID {parent_id} does not exist, cannot add subchannel.")
        if isinstance(indices, int):
            indices = [indices]
        for index in indices:
            if index < 0 or index >= self.channels[parent_id].shape[0]:
                raise ValueError(f"Index {index} is out of bounds for parent channel {parent_id} with {self.channels[parent_id].shape[0]} layers.")

        parent_channel = self.channels[parent_id]
        metadata['parent_channel'] = parent_channel
        metadata['parent_indices'] = indices
        metadata['parent_id'] = parent_id
        metadata['is_subchannel'] = True
        shape = (len(indices), *self.world_shape)
        init_func = lambda shp, md: (md['parent_channel'].contents[indices], md)
        self.channels[id] = Channel(id, shape, init_func, metadata, parent_channel.allowed_range)

    # TODO: add channels not of world shape and add associations to other world/other channels (for hierarchical channels)
    # EG: define a set of NN channels (and their genomes) that are associated with a world channel via a genome map

    def add_update_function(self, id, function, input_channel_ids: list[str] = [], affected_channel_ids: list[str] = [],
                            metadata: dict = None, req_sim_metadata: dict = {},
                            req_channel_metadata: dict = {}):
        if id in self.update_functions:
            raise ValueError(f"Update function with ID {id} already exists.")
        update_func = UpdateFunction(id, function, input_channel_ids, affected_channel_ids,
                                     metadata, req_sim_metadata, req_channel_metadata)
        update_func.assert_compatability(sim=self)
        self.update_functions[id] = update_func

    def init_all_channels(self):
        for channel in self.channels.values():
            try:
                if not channel.initialized:
                    if channel.metadata.get('is_subchannel', False):
                        channel.metadata['parent_channel'].init_contents()
                    channel.init_contents()
            except Exception as e:
                raise RuntimeError(f"An error occurred while initializing channel {channel.id}") from e

    def update(self):
        for _, update_function in self.update_functions.items():
            update_function.update(sim=self)

    # def update_group(self, old_group_id, new_group_id):

    # def define_channel_group(self, group_id, channel_ids: np.array, group_metadata, dtype=torch.float32, device=torch.device("cpu")):
    #     if group_id in self.channels:
    #         raise ValueError(f"Channel with ID {group_id} already exists.")

    #     flat_chids = set(channel_ids)
    #     subgroups = []
    #     chs_in_groups = set()
    #     for ch_id in channel_ids:
    #         if ch_id not in self.channels:
    #             raise ValueError(f"Channel with ID {ch_id} does not exist.")
    #         channel = self.channels[ch_id]
    #         if channel.metadata['is_group']:
    #             subgroups.append(ch_id)
    #             for ch_ 
    #             [flat_chids.add(grouped_ch_id) for grouped_ch_id in channel.metadata['channel_ids']]

    #     num_layers = sum([self.channels[ch_id].num_layers for ch_id in flat_chids])

    #     # Add group metadata
    #     additional_metadata = {
    #         'is_group': True,
    #         'num_layers': num_layers,     # for convenience
    #         'channel_ids': channel_ids,
    #         'flat_chids': flat_chids,
    #         'subgroup_ids': subgroups,
    #     }
    #     final_metadata = {**group_metadata, **additional_metadata}

    #     # Add the group tensor as a new Channel with appropriate metadata
    #     def init_group(shape, metadata):

    #         tensors = []
    #         for ch_id in metadata['flat_chids']:
    #             channel = self.channels[ch_id]
    #             if not channel.initialized:
    #                 channel.init_contents()
    #             tensors.append(channel.contents)
            
    #         group_tensor = torch.cat(tensors, dim=0)

    #         # redefine the base tensors in terms of this highest level (as of now) group
    #         start_index = 0
    #         for ch_id in metadata['flat_chids']:
    #             channel = self.channels[ch_id]
    #             del channel.contents
    #             num_layers = channel.num_layers
    #             channel.contents = group_tensor[start_index:start_index+num_layers].view(channel.num_layers, *self.world_shape)

    #         for gr_id in metadata['subgroups']:
    #             group_ch = self.channels[gr_id]
    #             del group_ch.contents
    #             group_ch.

    #         return group_tensor, {}
                

    #     self.add_channel(group_id, init_group, group_shape[0], metadata=final_metadata, dtype=dtype, device=device)