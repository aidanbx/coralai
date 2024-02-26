import torch
import taichi as ti
import torch.nn as nn

from ...dynamics.nn_lib import ch_norm
from ...dynamics.organism import Organism

@ti.data_oriented
class MinimalOrganism(Organism):
    def __init__(self, n_sensors, n_actuators, torch_device):
        super().__init__(n_sensors, n_actuators)

        # First convolutional layer
        self.conv = nn.Conv2d(
            self.n_sensors,
            self.n_actuators,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            device=torch_device,
            bias=False
        )

    
    def forward(self, x):
        with torch.no_grad():
            x = self.conv(x)
            x = nn.ReLU()(x)
            x = ch_norm(x)
            x = torch.sigmoid(x)

            return x


    def perturb_weights(self, perturbation_strength):
        self.conv.weight.data += perturbation_strength * torch.randn_like(self.conv.weight.data)
        