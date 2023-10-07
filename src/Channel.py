import numpy as np
import torch

class Channel:
    def __init__(self, id, shape, init_func, metadata: dict = None,
                 allowed_range = None, dtype = torch.float32, device = torch.device("cpu")):
        self.id = id
        self.shape = shape
        self.init_function = init_func

        if metadata is None:
            metadata = {
                'description': f'Channel id: {id}, shape: {shape}',
            }
        self.metadata = metadata

        self.allowed_range = allowed_range
        if allowed_range is None:
            allowed_range = (-np.inf,np.inf)

        assert 'description' in self.info, "info must contain 'description' key"

        self.dtype = dtype
        self.device = device
    
    def init(self):
        contents, init_metadata = self.init_function(self.shape, self.metadata)
        assert contents.shape == self.shape, f"init_function {id} must return tensor of shape {self.shape}"
        assert contents.shape[0] == self.num_components, f"init_function for {id} must return tensor with {self.num_components} components"
        assert contents.min() >= self.allowed_range[0], f"init_function for {id} must return tensor with min value >= {self.allowed_range[0]}"
        assert contents.max() <= self.allowed_range[1], f"init_function for {id} must return tensor with max value <= {self.allowed_range[1]}"
        
        self.metadata.update({"init_data": init_metadata})
        return contents