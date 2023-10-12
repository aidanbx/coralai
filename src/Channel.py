import numpy as np
import torch

class Channel:
    def __init__(self, id, shape, init_func=None, metadata: dict=None,
                 allowed_range=(-np.inf, np.inf), dtype=torch.float32, device=torch.device("cpu")):
        
        self.id = id
        self.shape = shape
        
        if init_func is None:
            init_func = lambda shape, md: (torch.zeros(shape, dtype=dtype, device=device), md)
        
        self.init_function = init_func
        self.allowed_range = allowed_range if allowed_range else (-np.inf, np.inf)
        
        default_metadata = {
            'id': id,
            'shape': self.shape,
            'allowed_range': self.allowed_range,
            'dtype': dtype,
            'device': device,
        }
        self.metadata = {**metadata, **default_metadata} if metadata else default_metadata

        self.dtype = dtype
        self.device = device
        self.initialized = False
    
    def init_contents(self):
        contents, init_metadata = self.init_function(self.shape, self.metadata)
        assert contents.shape == self.shape, f"init_function {self.id} must return tensor of shape {self.shape}, got {contents.shape}"
        assert contents.min() >= self.allowed_range[0], f"init_function for {self.id} must return tensor with min value >= {self.allowed_range[0]}, got {contents.min()}"
        assert contents.max() <= self.allowed_range[1], f"init_function for {self.id} must return tensor with max value <= {self.allowed_range[1]}, got {contents.max()}"
        
        self.metadata.update(init_metadata)
        self.contents = contents
        self.initialized=True
        return self.contents