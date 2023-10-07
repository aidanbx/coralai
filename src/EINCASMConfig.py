import yaml
import torch

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.dimensions = config.get('dimensions', 2)
        self.world_shape = config.get('world_shape', [100,100])
        self.kernel = None
        self.update_float_dtype(config['torch'].get('dtype', 'float32'))
        self.update_device(config['torch']['device']) # Refactor these two?
        self.VERBOSE = config.get('VERBOSE', False)
        
        self.update_kernel(config.get('kernel', None))
        self.env_config = config.get('environment', None)
        self.weather_config = config.get('weather', None)
        self.live_config = config.get('perception', None)
        self.physiological_config = config.get('physiology', None)

        self.num_communication_channels = config.get('num_communication_channels', 0)
        self.num_resource_types = config.get('num_resource_types', 0)

        self.num_muscles = self.kernel.shape[0] + 2 # Flow muscles for kernel + mine + resource muscles


    def update_kernel(self, kernel):
        if kernel is None:
            raise ValueError("No kernel specified")
        else:
            kernel = torch.tensor(kernel, device=self.device, dtype=torch.int8)
            assert kernel.shape[1] == self.dimensions, "Kernel must have dim(env) + 1 columns"
            assert torch.eq(kernel[0], self.zero_tensor.repeat(self.dimensions)).all(), "Kernel must have origin at index 0"
            assert kernel.shape[0] % 2 == 1, "Odd kernels (excluding origin) unimplemented"
            assert (torch.roll(kernel[1:], shifts=(kernel.shape[0]-1)//2, dims=0) - kernel[1:]).sum() == 0, "Kernel must be symmetric"
            self.kernel = kernel

    def get_config(self):
        return self.config

    def update_device(self, device):
        if device == 'mps':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                raise ValueError("MPS is not available on this system")
        elif device == 'cuda':
            self.device = torch.device("cuda")
        elif device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        if self.kernel is not None:
            self.kernel = self.kernel.to(device)
        self.zero_tensor = torch.tensor([0.0], device=self.device, dtype=self.float_dtype)
        self.one_tensor = torch.tensor([1.0], device=self.device, dtype=self.float_dtype)     

    def update_float_dtype(self, new_dtype):
        if new_dtype == 'float16':
            self.float_dtype = torch.float16
        elif new_dtype == 'float64':
            self.float_dtype = torch.float64
        elif new_dtype == 'float32':
            self.float_dtype = torch.float32
        else:
            raise ValueError(f"Unrecognized float dtype '{new_dtype}' specified")        

config = Config('config.yaml')