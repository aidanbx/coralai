import taichi as ti
import torch
import torch.nn as nn
from .nn_lib import ch_norm


@ti.data_oriented
class Organism(nn.Module):
    def __init__(self, world, sensors, n_actuators):
        super(Organism, self).__init__()
        self.world = world
        self.w = world.w
        self.h = world.h
        self.sensors = sensors
        self.sensor_inds = self.world.windex[self.sensors]
        self.n_sensors = self.sensor_inds.shape[0]
        self.n_actuators = n_actuators
        
        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_sensors,
            self.n_actuators,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=self.world.torch_device,
            bias=False
        )


    def forward(self, x=None):
        with torch.no_grad():
            if x is None:
                x=self.world.mem[:, self.sensor_inds, :, :]
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = ch_norm(x)
            return torch.sigmoid(x)


    def perturb_weights(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data) 
