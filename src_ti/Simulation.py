import taichi as ti
import numpy as np

class Channel:
    def __init__(self, dtype=ti.f32, init_func=None,
                 lims=None,
                 metadata: dict=None, **kwargs):
        
        self.lims = lims if lims else (-np.inf, np.inf)
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.init_func = init_func
        self.dtype = dtype

    def __getitem__(self, key):
        return self.metadata.get(key)

    def __setitem__(self, key, value):
        self.metadata[key] = value

class Rule:
    def __init__(self, id, func, metadata: dict=None, **kwargs):
        self.id = id
        self.func = func
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.metadata['id'] = self.id

    def apply(self, sim):
        self.func(sim)

@ti.data_oriented
class Simulation:
    def __init__(self, id, shape: tuple, channels: dict=None, metadata: dict = None, **kwargs):
        self.id = id
        self.shape = shape
        self.data = None
        self.metadata = metadata if metadata is not None else {}
        self.metadata.update(kwargs)
        self.metadata['id'] = self.id
        self.channel_ids = [] 

        if channels is not None:
            self.add_channels(channels)

    def add_channels(self, channels: dict):
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, Channel):
                 self.__setitem__(id, ch)
            elif isinstance(ch, dict):
                self.channel(chid, **ch)
            else:
                self.channel(chid, ch)


    def channel(self, id: str, dtype=ti.f32, **kwargs):
        if self.data is not None:
            raise ValueError("Cannot add channel after world is defined (yet)")
        self.__setitem__(id, Channel(**kwargs))
    
    def init_data(self):
        self.data = ti.Struct.field(
            {chid: self[chid].dtype for chid in self.channel_ids},
            shape=self.shape)

    def __getitem__(self, key):
        return self.metadata.get(key)

    def __setitem__(self, key, value):
        if isinstance(value, Channel):
            self.metadata[key] = value
            self.channel_ids.append(key)
        self.metadata[key] = value
     
