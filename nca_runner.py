import os
import torch
import taichi as ti
import torch.nn as nn

from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import Visualization
from coralai.evolution.hyper_organism import HyperOrganism
from coralai.evolution.ecosystem import Ecosystem

class CoralVis(Visualization):
    def __init__(self, substrate, ecosystem, vis_chs):
        super().__init__(substrate, vis_chs)
        self.ecosystem = ecosystem

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
            sub_w.text(
                # f"Energy: {self.substrate.mem[0, inds.energy, pos_x, pos_y]:.2f}," +
                # f"Infra: {self.substrate.mem[0, inds.infra, pos_x, pos_y]:.2f}," +
                f"Genome: {self.substrate.mem[0, inds.genome, pos_x, pos_y]:.2f}, " 
                # f"Acts: {self.substrate.mem[0, inds.acts, pos_x, pos_y]}"
            )
            sub_w.text(f"Population:")
            for genome_key in self.ecosystem.population.keys():
                sub_w.text(f"{genome_key}: {self.ecosystem.population[genome_key]['infra']}")
            # for channel_name in ['energy', 'infra']:
            #     chindex = self.world.windex[channel_name]
            #     max_val = self.world.mem[0, chindex].max()
            #     min_val = self.world.mem[0, chindex].min()
            #     avg_val = self.world.mem[0, chindex].mean()
            #     sum_val = self.world.mem[0, chindex].sum()
            #     sub_w.text(f"{channel_name}: Max: {max_val:.2f}, Min: {min_val:.2f}, Avg: {avg_val:.2f}, Sum: {sum_val:.2f}")


def nca_activation(mem):
    mem = nn.ReLU()(mem)
    # Calculate the mean across batch and channel dimensions
    mean = mem.mean(dim=(0, 2, 3), keepdim=True)
    # Calculate the variance across batch and channel dimensions
    var = mem.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    # Normalize the input tensor
    mem.sub_(mean).div_(torch.sqrt(var + 1e-5))
    mem = torch.sigmoid(mem)
    return mem


def main(config_filename, channels, shape, kernel, sense_chs, act_chs, torch_device):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()
    inds = substrate.ti_indices[None]

    num_cells_to_activate = 10
    for _ in range(num_cells_to_activate):
        x = torch.randint(0, shape[0], (1,))
        y = torch.randint(0, shape[1], (1,))
        substrate.mem[0,inds.genome, x, y] = 1

    def _create_organism(genome_key, genome=None):
        org = HyperOrganism(config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        if genome is None:
            genome = org.gen_random_genome(genome_key)
        org.set_genome(genome_key, genome=genome)
        org.create_torch_net()
        return org
    
    def _apply_physics():
        substrate.mem[:, ecosystem.act_chinds] = nca_activation(substrate.mem[:, ecosystem.act_chinds])

    ecosystem = Ecosystem(substrate, _create_organism, _apply_physics, min_size = 1)
    vis = CoralVis(substrate, ecosystem, [('rgb', 'r'), ('rgb', 'g'), ('rgb', 'b')])
    # out_mem = torch.zeros_like(substrate.mem[0, organism.act_chinds])
    genome_key = 0
    ecosystem.population[genome_key]['infra'] = 1000 # to keep alive
    while vis.window.running:
        # for _ in range(20):
        #     matches = substrate.mem[0,0].eq(0)
        #     coords = torch.where(matches)
        #     if coords[0].shape[0] != 0:
        #         pass
        #     combined_coords = torch.stack((coords[0], coords[1]), dim=1).contiguous()

        substrate.mem[0, inds.rgb] += torch.rand_like(substrate.mem[0, inds.rgb]) * 0.1
        ecosystem.update()
        vis.update()
        if vis.mutating:
            new_genome_key = ecosystem.mutate(genome_key, report=True)
            ecosystem.population[genome_key]['infra'] = 0
            ecosystem.population[new_genome_key]['infra'] = 1000 # to keep alive
            genome_key = new_genome_key
            substrate.mem[0, inds.genome,...] = genome_key
            vis.mutating=False


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename="coralai/instances/nca/nca_neat.config",
        channels={
            "rgb": ti.types.struct(r=ti.f32, g=ti.f32, b=ti.f32),
            "hidden": ti.types.vector(n=2, dtype=ti.f32),
            "genome": ti.f32,
        },
        shape=(80, 80),
        kernel=[        [0,-1],
                [-1, 0],[0, 0],[1, 0],
                        [0, 1]],
        sense_chs=['rgb', 'hidden'],
        act_chs=['rgb', 'hidden'],
        torch_device=torch_device
    )
