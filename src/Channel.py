import numpy as np
import torch
import importlib
import src.Simulation as Simulation
importlib.reload(Simulation)

class Channel:
    def __init__(self, id, shape, init_func, metadata: dict = None,
                 allowed_range = (-np.inf,np.inf), dtype = torch.float32, device = torch.device("cpu")):
        self.id = id
        self.shape = shape
        self.init_function = init_func
        if allowed_range is None:
            allowed_range = (-np.inf,np.inf)
        self.allowed_range = allowed_range
        default_metadata = {
            'id': id,
            'shape': shape,
            'allowed_range': self.allowed_range,
            'dtype': dtype,
            'device': device,
        }
        if metadata is None:
            metadata = {
                'description': f'Channel id: {id}, shape: {shape}',
            }
        self.metadata = metadata
        self.metadata.update(default_metadata)

        assert 'description' in self.metadata.keys(), "Metadata must contain \"description\" key"

        self.dtype = dtype
        self.device = device
    
    def init_contents(self):
        contents, init_metadata = self.init_function(self.shape, self.metadata)
        assert contents.shape == self.shape, f"init_function {self.id} must return tensor of shape {self.shape}"
        assert contents.min() >= self.allowed_range[0], f"init_function for {self.id} must return tensor with min value >= {self.allowed_range[0]}"
        assert contents.max() <= self.allowed_range[1], f"init_function for {self.id} must return tensor with max value <= {self.allowed_range[1]}"
        
        self.metadata.update(init_metadata)
        self.contents = contents
        return self.contents