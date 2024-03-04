import os
import torch
import taichi as ti
from coralai.substrate.substrate import Substrate
from coralai.instances.coral.coral_physics import apply_actuators
from coralai.evolution.ecosystem import Ecosystem
from coralai.substrate.visualization import Visualization
from coralai.instances.coral.coral_organism_cnn import CoralOrganism
from coralai.evolution.hyper_organism import HyperOrganism

class CoralVis(Visualization):
    def __init__(self, substrate, vis_chs):
        super().__init__(substrate, vis_chs)

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(240 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.opt_window(sub_w)
            current_pos = self.window.get_cursor_pos()
            pos_x = int(current_pos[0] * self.w) % self.w
            pos_y = int(current_pos[1] * self.h) % self.h
            sub_w.text(f"Stats at ({pos_x}, {pos_y}):")
            sub_w.text(f"Energy: {self.substrate.mem[0, inds.energy, pos_x, pos_y]}, Infra: {self.substrate.mem[0, inds.infra, pos_x, pos_y]}")
            # for channel_name in ['energy', 'infra']:
            #     chindex = self.world.windex[channel_name]
            #     max_val = self.world.mem[0, chindex].max()
            #     min_val = self.world.mem[0, chindex].min()
            #     avg_val = self.world.mem[0, chindex].mean()
            #     sum_val = self.world.mem[0, chindex].sum()
            #     sub_w.text(f"{channel_name}: Max: {max_val:.2f}, Min: {min_val:.2f}, Avg: {avg_val:.2f}, Sum: {sum_val:.2f}")


def main(config_filename, channels, shape, kernel, sense_chs, act_chs, torch_device):
    kernel = torch.tensor(kernel, device=torch_device)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()
    vis = CoralVis(substrate, ['energy', 'infra', 'energy'])

    def create_organism(genome_key):
        org = HyperOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        org.set_genome(genome_key, genome=org.gen_random_genome())
        org.create_torch_net()
        return org
    
    ecosystem = Ecosystem(substrate, create_organism, 3, 3)
    # initialize infrastructure with random values
    # rand_val = torch.randn_like(substrate.mem[0, inds.infra])
    # substrate.mem[0, inds.infra] = torch.where(rand_val>3, rand_val, 0)


    while vis.window.running:
        vis.update()
        apply_actuators(substrate, ecosystem, kernel)
        

if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename = "coralai/instances/coral/coral_neat.config",
        channels = {
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=7, dtype=ti.f32) # must equal length of kernel
            ),
            "com": ti.types.vector(n=8, dtype=ti.f32),
            "genome": ti.f32,
        },
        shape = (80,80),
        kernel = [[-1,-1],[0,-1],
                  [-1, 0],[0, 0],[1, 0],
                          [0, 1],[1, 1]],
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )
