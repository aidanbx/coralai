import torch
import taichi as ti
import numpy as np
import warnings

class Channel:
    def __init__(
            self, id, world, ti_dtype=ti.f32,
            lims=None,
            metadata: dict=None, **kwargs):
        self.id = id
        self.world = world
        self.lims = np.array(lims) if lims else np.array([-1, 1])
        self.ti_dtype = ti_dtype
        self.memblock = None
        self.indices = None
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        field_md = {
            'id': self.id,
            'ti_dtype': self.ti_dtype,
            'lims': self.lims,
        }
        self.metadata.update(field_md)
    
    def link_to_mem(self, indices, memblock):
        self.memblock = memblock
        indices = np.array(indices)
        if len(indices) == 1:
            indices = indices[0]
        self.indices = indices
        self.metadata['indices'] = indices
    
    def add_subchannel(self, id, ti_dtype=ti.f32, **kwargs):
        subch = Channel(id, self.world, ti_dtype=ti_dtype, **kwargs)
        subch.metadata['parent'] = self
        self.metadata[id] = subch
        self.metadata['subchids'] = self.metadata.get('subchids', [])
        self.metadata['subchids'].append(id)
        return subch

    def get_data(self):
        if self.memblock is None:
            raise ValueError(f"Channel: Channel {self.id} has not been allocated yet.")
        else:
            return self.memblock[self.indices]

    def __getitem__(self, key):
        return self.metadata.get(key)

    def __setitem__(self, key, value):
        self.metadata[key] = value
            
@ti.data_oriented
class World:
    # TODO: Support multi-level indexing beyond 2 levels
    # TODO: Support mixed taichi and torch tensors - which will be transferred more?
    def __init__(self, shape, torch_dtype, torch_device, channels: dict=None):
        self.shape = (*shape, 0)
        self.mem = None
        self.indices = None
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device
        self.channels = {}
        if channels is not None:
            self.add_channels(channels)
        self._ti_inds_dict = {}
        self._ti_inds_types = {}

    def add_channel(self, id: str, ti_dtype=ti.f32, **kwargs):
        if self.mem is not None:
            raise ValueError(f"World: When adding channel {id}: Cannot add channel after world memory is allocated (yet).")
        self.channels[id] = Channel(id, self, ti_dtype=ti_dtype, **kwargs)

    def add_channels(self, channels: dict):
        if self.mem is not None:
            raise ValueError(f"World: When adding channels {channels}: Cannot add channels after world memory is allocated (yet).")
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, dict):
                self.add_channel(chid, **ch)
            else:
                self.add_channel(chid, ch)
        
    def check_ch_shape(self, shape):
        lshape = len(shape)
        if lshape > 3 or lshape < 2:
            raise ValueError(f"World: Channel shape must be 2 or 3 dimensional. Got shape: {shape}")
        if shape[:2] != self.shape[:2]:
            print(shape[:2], self.shape[:2])
            raise ValueError(f"World: Channel shape must be (w, h, ...) where w and h are the world dimensions: {self.shape}. Got shape: {shape}")
        if lshape == 2:
            return 1
        else:
            return shape[2]

    def _transfer_to_mem(self, mem, tensor_dict, index_tree, channel_dict):
        for chid, chindices in index_tree.items():
            if 'subchannels' in chindices:
                for subchid, subchtree in chindices['subchannels'].items():
                    if tensor_dict[chid][subchid].dtype != self.torch_dtype:
                        warnings.warn(f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m", stacklevel=3)
                    if len(tensor_dict[chid][subchid].shape) == 2:
                        tensor_dict[chid][subchid] = tensor_dict[chid][subchid].unsqueeze(2)
                    mem[:, :, subchtree['indices']] = tensor_dict[chid][subchid].type(self.torch_dtype)
                    channel_dict[chid].add_subchannel(subchid, ti_dtype=channel_dict[chid].ti_dtype)
                    channel_dict[chid][subchid].link_to_mem(subchtree['indices'], mem)
                channel_dict[chid].link_to_mem(chindices['indices'], mem)
            else:
                if tensor_dict[chid].dtype != self.torch_dtype:
                    warnings.warn(f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m", stacklevel=3)
                if len(tensor_dict[chid].shape) == 2:
                    tensor_dict[chid] = tensor_dict[chid].unsqueeze(2)
                mem[:, :, chindices['indices']] = tensor_dict[chid].type(self.torch_dtype)
                channel_dict[chid].link_to_mem(chindices['indices'], mem)
        return mem, channel_dict

    def add_ti_inds(self, chid, inds):
        inds=np.array(inds, dtype=np.int32)
        inds_vec = ti.Vector(inds)
        inds_vec_type = ti.types.vector(n=inds.shape[0], dtype=ti.i32)
        self._ti_inds_types[chid] = inds_vec_type
        self._ti_inds_dict[chid] = inds_vec
        
    def _index_subchannels(self, subchdict, start_ind, parent_chid):
        end_ind = start_ind
        subch_tree = {}
        for subchid, subch in subchdict.items():
            if not isinstance(subch, torch.Tensor):
                raise ValueError(f"World: Channel grouping only supported up to a depth of 2. Subchannel {subchid} of channel {parent_chid} must be a torch.Tensor. Got type: {type(subch)}")
            subch_depth = self.check_ch_shape(subch.shape)
            inds = [i for i in range(end_ind, end_ind+subch_depth)]
            self.add_ti_inds(parent_chid + "_" + subchid, inds)
            subch_tree[subchid] = {
                'indices': inds,
            }
            end_ind += subch_depth
        return subch_tree, end_ind-start_ind

    def malloc(self):
        if self.mem is not None:
            raise ValueError(f"World: Cannot allocate world memory twice.")
        celltype = ti.types.struct(**{chid: self.channels[chid].ti_dtype for chid in self.channels.keys()})
        tensor_dict = celltype.field(shape=self.shape[:2]).to_torch(device=self.torch_device)

        index_tree = {}
        endlayer_pointer = self.shape[2]
        for chid, chdata in tensor_dict.items():
            if isinstance(chdata, torch.Tensor):
                ch_depth = self.check_ch_shape(chdata.shape)
                inds = [i for i in range(endlayer_pointer, endlayer_pointer + ch_depth)]
                self.add_ti_inds(chid, inds)
                index_tree[chid] = {'indices': inds}
                endlayer_pointer += ch_depth
            elif isinstance(chdata, dict):
                subch_tree, total_depth = self._index_subchannels(chdata, endlayer_pointer, chid)
                inds = [i for i in range(endlayer_pointer, endlayer_pointer + total_depth)]
                self.add_ti_inds(chid, inds)
                index_tree[chid] = {
                    'subchannels': subch_tree,
                    'indices': inds,
                }
                endlayer_pointer += total_depth
                
        mem = torch.empty((*self.shape[:2], endlayer_pointer), dtype=self.torch_dtype, device=self.torch_device)
        self.mem, self.channels = self._transfer_to_mem(mem, tensor_dict, index_tree, self.channels)
        # self.mem = self.mem.permute(2, 0, 1)
        # self.shape = self.mem.shape
        del tensor_dict
        self.indices = self._windex(index_tree)
        ti_inds_struct_type = ti.types.struct(**self._ti_inds_types)
        ti_inds_field = ti_inds_struct_type.field(shape=())
        for chid, inds in self._ti_inds_dict.items():
            ti_inds_field[None][chid] = inds
        self.ti_inds = ti_inds_field[None]
    
    def __getitem__(self, key):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot get {key}")
        val = self.mem[:, :, self.indices[key]]
        return val

    def __setitem__(self, key, value):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot set {key}")
        indices = self.indices[key]
        if len(indices) > 1:
            if value.shape != self[key].shape:
                raise ValueError(f"World: Cannot set channel(s) {key} to value of shape {value.shape}. Expected shape: {self[key].shape}")
            self.mem[:, :, indices].copy_(value)
        if len(indices) == 1:
            if len(value.shape) == 3:
                value = value.squeeze(2)
            if value.shape != self.shape[:2]:
                raise ValueError(f"World: Cannot set channel {key} to value of shape {value.shape}. Expected shape: {self.shape[:2]}")
            self.mem[:, :, indices[0]].copy_(value)

    class _windex:
        def __init__(self, index_tree):
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
                return np.array(self._get_tuple_inds(key))
            
            elif isinstance(key, list):
                inds = []
                for chid in key:
                    if isinstance(chid, tuple):
                        inds += self._get_tuple_inds(chid)
                    else:
                        inds += self.index_tree[chid]['indices']
                return np.array(inds)
            else:
                return np.array(self.index_tree[key]['indices'])
        
        def __setitem__(self, key, value):
            raise ValueError(f"World: World indices are read-only. Cannot set index {key} to {value} - get/set to the world iteself")
    