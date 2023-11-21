import taichi as ti
import torch

from eincasm_python.nca import NCA
from eincasm_python.analysis.vis_slim import Vis

ti.init(ti.metal)

ti.init(ti.gpu)  
ein = NCA(shape=(400, 400), torch_device=torch.device("mps"))

w = ein.world

vis = Vis(w, [('com', 'r'), ('com', 'g'), ('com', 'b')])

while vis.window.running:
    if vis.perturbing_weights:
        ein.organism.perturb_weights(vis.perturbation_strength)
    ein.world.mem = ein.organism.forward(ein.world.mem)
    vis.update()
