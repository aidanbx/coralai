import torch
import taichi as ti
import numpy as np
import warnings
from collections import deque
            
# TODO: Support true multi-level indexing by creating a contiguous world tensor with indexing info
# -- This is hard because the cell of subchannels can be matrices/other tensors
class World:
    def __init__(self, shape, dtype, torch_device, channels: dict=None):
        self.shape = (*shape, 0)
        self.dtype = dtype
        self.torch_device = torch_device
        self.channels = {}
        self.memory_allocated = False
        if channels is not None:
            self.add_channels(channels)
        self.tensor_dict = None
        self.mem = None
        self.data = None

    def add_channel(self, id: str, dtype=ti.f32, **kwargs):
        if self.memory_allocated:
            raise ValueError("When adding channel {id}: Cannot add channel after world memory is allocated (yet).")
        self.channels[id] = Channel(id=id, dtype=dtype, **kwargs)

    def add_channels(self, channels: dict):
        if self.memory_allocated:
            raise ValueError("When adding channels {channels}: Cannot add channels after world memory is allocated (yet).")
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, Channel):
                 if ch.id is None:
                     ch.id = chid
                 self.channels[id] = ch
            elif isinstance(ch, dict):
                self.add_channel(chid, **ch)
            else:
                self.add_channel(chid, ch)
        
    def check_ch_shape(self, shape):
        lshape = len(shape)
        if lshape > 3 or lshape < 2:
            raise ValueError(f"Channel shape must be 2 or 3 dimensional. Got shape: {shape}")
        if shape[:2] != self.shape[:2]:
            print(shape[:2], self.shape[:2])
            raise ValueError(f"Channel shape must be (w, h, ...) where w and h are the world dimensions: {self.shape}. Got shape: {shape}")
        if lshape == 2:
            return 1
        else:
            return shape[2]

    def _transfer_to_mem(self, mem, tensor_dict, index_tree, channel_dict):
        for chid, chindices in index_tree.items():
            if 'subchannels' in chindices:
                for subchid, subchtree in chindices['subchannels'].items():
                    if tensor_dict[chid][subchid].dtype != self.dtype:
                        warnings.warn(f"Warning: The dtype of subchannel[{chid}][{subchid}] ({tensor_dict[chid][subchid].dtype}) does not match the dtype of its world. Casting to {self.dtype}.")
                    if len(tensor_dict[chid][subchid].shape) == 2:
                        tensor_dict[chid][subchid] = tensor_dict[chid][subchid].unsqueeze(2)
                    mem[:, :, subchtree['indices']] = tensor_dict[chid][subchid].type(self.dtype)
                    channel_dict[chid].add_subchannel(subchid, dtype=self.dtype)
                    channel_dict[chid][subchid].index(subchtree['indices'], mem)
                channel_dict[chid].index(chindices['indices'], mem)
            else:
                if tensor_dict[chid].dtype != self.dtype:
                    warnings.warn(f"Warning: The dtype of channel[{chid}] ({tensor_dict[chid].dtype}) does not match the dtype of its world. Casting to {self.dtype}.")
                if len(tensor_dict[chid].shape) == 2:
                    tensor_dict[chid] = tensor_dict[chid].unsqueeze(2)
                mem[:, :, chindices['indices']] = tensor_dict[chid].type(self.dtype)
                channel_dict[chid].index(chindices['indices'], mem)
        return mem, channel_dict
    
    def _index_subchannels(self, subchdict, start_ind, parent_chid):
        end_ind = start_ind
        subch_tree = {}
        for subchid, subch in subchdict.items():
            if not isinstance(subch, torch.Tensor):
                raise ValueError(f"Channel grouping only supported up to a depth of 2. Subchannel {subchid} of channel {parent_chid} must be a torch.Tensor. Got type: {type(subch)}")
            subch_depth = self.check_ch_shape(subch.shape)
            subch_tree[subchid] = {
                'indices': [i for i in range(end_ind, end_ind+subch_depth)]
            }
            end_ind += subch_depth
        return subch_tree, end_ind-start_ind
    
    def malloc(self):
        self.celltype = ti.types.struct(**{chid: self.channels[chid].dtype for chid in self.channels.keys()})
        tensor_dict = self.celltype.field(shape=self.shape[:2]).to_torch(device=self.torch_device)

        index_tree = {}
        endlayer_pointer = self.shape[2]
        for chid, chdata in tensor_dict.items():
            if isinstance(chdata, torch.Tensor):
                ch_depth = self.check_ch_shape(chdata.shape)
                index_tree[chid] = {'indices': [i for i in range(endlayer_pointer, endlayer_pointer + ch_depth)]}
                endlayer_pointer += ch_depth
            elif isinstance(chdata, dict):
                subch_tree, total_depth = self._index_subchannels(chdata, endlayer_pointer, chid)
                index_tree[chid] = {
                    'subchannels': subch_tree,
                    'indices': [i for i in range(endlayer_pointer, endlayer_pointer + total_depth)]
                }
                endlayer_pointer += total_depth
                
        self.shape = (*self.shape[:2], endlayer_pointer)
        data = torch.empty(self.shape, dtype=torch.float32, device=self.torch_device)
        self.mem, self.channels = self._transfer_to_mem(data, tensor_dict, index_tree, self.channels)
        del tensor_dict
        self.memory_allocated = True
        self.data = self._worldata(self.mem, index_tree)
        return self.mem, index_tree
    
    def __getitem__(self, key):
        return self.channels.get(key)
    
    class _worldata:
        def __init__(self, mem, index_tree):
            self.mem = mem
            self.index_tree = index_tree

        def _get_tuple_inds(self, key_tuple):
            chid = key_tuple[0]
            subchid = key_tuple[1]
            if isinstance(subchid, list):
                inds = []
                for subchid_single in key_tuple[1]:
                    inds += self.index_tree[chid]['subchannels'][subchid_single]['indices']
            else:
                inds = self.index_tree[chid]['subchannels'][subchid]['indices']
            return inds

        def __getitem__(self, key):
            if isinstance(key, tuple):
                return self.mem[:, :, self._get_tuple_inds(key)]
            elif isinstance(key, list):
                inds = []
                for chid in key:
                    if isinstance(chid, tuple):
                        inds += self._get_tuple_inds(chid)
                    else:
                        inds += self.index_tree[chid]['indices']
                return self.mem[:, :, inds]
            else:
                return self.mem[:, :, self.index_tree[key]['indices']]
        
        def __setitem__(self, key, value):
            raise ValueError("Cannot set world data directly. Use world.add_channels() or world.add_channel() to add channels to the world.")
            # if isinstance(key, tuple):
            #     chid = key[0]
            #     subchid = key[1]
            #     self.mem[:,:,self.index_tree[chid]['subchannels'][subchid]['indices']] = value
            # else:
            #     self.mem[self.index_tree[key]['indices']] = value

    def __setitem__(self, key, value):
        if self.mem is not None:
            raise ValueError("Cannot add channels after world memory is allocated (yet).")
        else:
            self.add_channels({key: value})
    
class Channel:
    def __init__(
            self, id=None, dtype=ti.f32,
            arch_device: tuple = (ti.cpu, torch.device("cpu")),
            init_func=None,
            lims=None,
            metadata: dict=None, **kwargs):
        self.id = id
        self.ti_arch = arch_device[0]
        self.torch_device = arch_device[1]
        self.lims = lims if lims else (-np.inf, np.inf)
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.init_func = init_func
        self.dtype = dtype
        self.memblock = None
        self.indices = None
    
    def index(self, indices, memblock):
        self.memblock = memblock
        if len(indices) == 1:
            indices = indices[0]
        self.indices = indices
        self.metadata['indices'] = indices
    
    def add_subchannel(self, id, dtype=ti.f32, **kwargs):
        subch = Channel(id=id, dtype=dtype, **kwargs)
        subch.metadata['parent'] = self
        self.metadata[id] = subch
        self.metadata['subchids'] = self.metadata.get('subchids', [])
        self.metadata['subchids'].append(id)
        return subch

    def data(self):
        if self.memblock is None:
            raise ValueError(f"Channel {self.id} has not been allocated yet.")
        else:
            return self.memblock[self.indices]

    def __getitem__(self, key):
        return self.metadata.get(key)

    def __setitem__(self, key, value):
        self.metadata[key] = value