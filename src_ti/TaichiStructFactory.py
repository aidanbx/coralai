import numpy as np
import taichi as ti
import os
os.environ['TI_WARN_ON_TYPE_CAST'] = '0'

class TaichiStructFactory:
    def __init__(self):
        self.val_dict = {}
        self.type_dict = {}

    def add(self, name, val, dtype):
        self.val_dict[name] = val
        self.type_dict[name] = dtype
    
    def add_nparr_float(self, name, val):
        val = np.array(val, dtype=np.float32)
        self.val_dict[name] = val
        self.type_dict[name] = ti.types.vector(n=val.shape[0], dtype=ti.f32)

    def add_nparr_int(self, name, val):
        val = np.array(val, dtype=np.int32)
        self.val_dict[name] = val
        self.type_dict[name] = ti.types.vector(n=val.shape[0], dtype=ti.i32)
    
    def add_ti(self, name, val):
        self.val_dict[name] = val
        self.type_dict[name] = type(val)

    def build(self):
        struct_type = ti.types.struct(**self.type_dict)
        val_field = struct_type.field(shape=())
        for k, v in self.val_dict.items():
            val_field[None][k] = v
        return val_field[None]
    