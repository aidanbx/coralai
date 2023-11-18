import taichi as ti
import torch
from eincasm_python.eincasm import Eincasm
from eincasm_python.analysis.vis import Vis

ti.init(ti.gpu)  
ein = Eincasm(shape=(10, 10), torch_device=torch.device("mps"), num_com=5)

w = ein.world

vis = Vis(w, ['capital', ('com', 'g'), 'total_mass'])
vis2 = Vis(w, [('com', 'r'), ('com', 'g'), ('com', 'b')])

while vis.window.running or vis2.window.running:
    vps = vis.params[None]
    vps2 = vis2.params[None]
    if vps.is_perturbing_weights or vps2.is_perturbing_weights:
        ein.organism.perturb_weights(max(vps.perturb_strength, vps2.perturb_strength))
    if vps.is_perturbing_biases or vps2.is_perturbing_biases:
        ein.organism.perturb_biases(max(vps.perturb_strength, vps2.perturb_strength))
    if not vps.drawing and not vps2.drawing:
        ein.apply_physics()
        ein.organism.apply_weights()
    vis.update()
    vis2.update()