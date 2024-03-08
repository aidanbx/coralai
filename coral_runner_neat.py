import os
import torch
import taichi as ti
from coralai.instances.coral.coral_physics import apply_physics
from coralai.substrate.substrate import Substrate
from coralai.evolution.neat_evolver import NEATEvolver
from coralai.substrate.visualization import Visualization

class CoralVis(Visualization):
    def __init__(self, substrate, ecosystem, vis_chs):
        super().__init__(substrate, vis_chs)
        self.ecosystem = ecosystem

    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(340 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.opt_window(sub_w)
            current_pos = self.window.get_cursor_pos()
            pos_x = int(current_pos[0] * self.w) % self.w
            pos_y = int(current_pos[1] * self.h) % self.h
            sub_w.text(f"Stats at ({pos_x}, {pos_y}):")
            sub_w.text(
                f"Energy: {self.substrate.mem[0, inds.energy, pos_x, pos_y]:.2f}," +
                f"Infra: {self.substrate.mem[0, inds.infra, pos_x, pos_y]:.2f}," +
                f"Genome: {self.substrate.mem[0, inds.genome, pos_x, pos_y]:.2f}, " 
                # f"Acts: {self.substrate.mem[0, inds.acts, pos_x, pos_y]}"
            )
            sub_w.text(f"Total Energy: {torch.sum(self.substrate.mem[0, inds.energy])}")
            sub_w.text(f"Energy Offset: {self.ecosystem.energy_offset}")


def main(config_filename, channels, shape, kernel, ind_of_middle, sense_chs, act_chs, torch_device):
    kernel = torch.tensor(kernel, device=torch_device)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()

    neat_evolver = NEATEvolver(config_path, substrate, kernel, ind_of_middle, sense_chs, act_chs,)
    
    def eval_vis(genomes, config):
        vis = CoralVis(substrate, neat_evolver, ["energy", "infra", "genome"])
        import random
        random_steps = random.randint(100, 500)
        neat_evolver.eval_genomes(genomes, config, random_steps, vis)
    neat_evolver.population.run(eval_vis, 100)


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename = "coralai/instances/coral/coral_neat.config",
        channels = {
            "genome": ti.f32,
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=5, dtype=ti.f32) # must equal length of kernel
            ),
            "com": ti.types.struct(
                a=ti.f32,
                b=ti.f32,
                c=ti.f32,
                d=ti.f32
            ),
        },
        shape = (200, 200),
        kernel = [        [0,-1],
                  [-1, 0],[0, 0],[1, 0],
                          [0, 1],        ],
        ind_of_middle = 2,
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )
