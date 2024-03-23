import taichi as ti
import torch

@ti.func
def ReLU(x):
    return x if x > 0 else 0

@ti.func
def sigmoid(x):
    return 1 / (1 + ti.exp(-x))

@ti.func
def inverse_gaussian(x):
    return -1./(ti.exp(0.89*ti.pow(x, 2.))+1.)+1.


def ch_norm(input_tensor):
    # Calculate the mean across batch and channel dimensions
    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)

    # Calculate the variance across batch and channel dimensions
    var = input_tensor.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

    # Normalize the input tensor
    input_tensor.sub_(mean).div_(torch.sqrt(var + 1e-5))

    return input_tensor

class TaichiStructFactory:
    """This class is a factory for creating Taichi structures."""

    def __init__(self):
        self.val_dict = {}
        self.type_dict = {}
        self.struct_type = None

    def add(self, name, val, dtype):
        self.val_dict[name] = val
        self.type_dict[name] = dtype

    def add_i(self, name, val):
        self.add(name, val, ti.i32)

    def add_f(self, name, val):
        self.add(name, val, ti.f32)

    def add_nparr_float(self, name, val):
        self.add(name, val, ti.types.vector(n=val.shape[0], dtype=ti.f32))

    def add_nparr_int(self, name, val):
        self.add(name, val, ti.types.vector(n=val.shape[0], dtype=ti.i32))

    def add_tivec_f(self, name, val):
        self.add(name, val, ti.types.vector(n=val.n, dtype=ti.f32))

    def add_tivec_i(self, name, val):
        self.add(name, val, ti.types.vector(n=val.n, dtype=ti.i32))

    def add_timat_f(self, name, val):
        self.add(name, val, ti.types.matrix(n=val.n, m=val.m, dtype=ti.f32))

    def add_timat_i(self, name, val):
        self.add(name, val, ti.types.matrix(n=val.n, m=val.m, dtype=ti.i32))

    def build(self):
        struct_type = ti.types.struct(**self.type_dict)
        val_field = struct_type.field(shape=())
        for k, v in self.val_dict.items():
            val_field[None][k] = v
        self.struct_type = struct_type
        return val_field