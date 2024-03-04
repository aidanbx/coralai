import os
import torch
import neat
import taichi as ti
import configparser
from datetime import datetime
import torch.nn as nn

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import relu_activation, sigmoid_activation, tanh_activation, identity_activation
from pytorch_neat.linear_net import LinearNet
from ...evolution.neat_organism import NeatOrganism
from ...substrate.nn_lib import ch_norm


@ti.data_oriented
class NCAOrganismHyper(NeatOrganism):
    def __init__(self, neat_config_path, substrate, kernel, sense_chs, act_chs, torch_device):
        super().__init__(neat_config_path, substrate, kernel, sense_chs, act_chs, torch_device)
        self.name = "NCA_HyperNEAT"
        self.net = None
        self.neat_config = self.load_neat_config()
        self.w = self.substrate.w
        self.h = self.substrate.h


    def load_neat_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        # Adaptive Linear Net:
        # ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
        # ["delta_w"],

        # Adaptive Net
        # ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],
        # ['w_ih', 'b_h', 'w_hh', 'b_o', 'w_ho', 'delta_w'])

        # Linear Net
        # ["x_in", "y_in", "z_in", "x_out", "y_out", "z_out"],
        # ["w, b"],
        n_in = 6
        n_out = 2
        genome_section = 'DefaultGenome'
        config.set(genome_section, 'num_inputs', f'{n_in}')
        config.set(genome_section, 'num_hidden', '7')
        config.set(genome_section, 'num_outputs', f'{n_out}')

        # Save the modified configuration in 'configs' folder with a specific name format
        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        config_dir = f'history/{self.name}'
        os.makedirs(config_dir, exist_ok=True)
        temp_config_path = os.path.join(config_dir, f'config_{current_datetime}.ini')
        with open(temp_config_path, 'w') as config_file:
            config.write(config_file)

        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           temp_config_path)


    def create_torch_net(self, batch_size = None):
        if batch_size is None:
            batch_size=self.substrate.w*self.substrate.h

        input_coords = []
        for offset in self.kernel:
            for ch in range(self.n_senses):
                input_coords.append([offset[0], offset[1], self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])
        

        self.net = LinearNet.create(
            self.genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.0,
            weight_max=3.0,
            batch_size=batch_size,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.torch_device,
        )
        return self.net

    # @ti.kernel
    # def mm_weights(self, mem: ti.types.ndarray(), weights: ti.types.ndarray(),
    #              cell_coords: ti.types.ndarray(), kernel: ti.types.ndarray(),
    #              sense_chinds: ti.types.ndarray(), act_chinds: ti.types.ndarray()):
    #     for cell_i, neigh_i, sense_i, act_i in ti.ndrange(cell_coords.shape[0], kernel.shape[0],
    #                                                     sense_chinds.shape[0], act_chinds.shape[0]):
    #         center_x = cell_coords[cell_i, 0]
    #         center_y = cell_coords[cell_i, 1]
    #         neigh_x = center_x + kernel[neigh_i, 0]
    #         neigh_y = center_y + kernel[neigh_i, 1]
    #         sensor_val = mem[0, sense_chinds[sense_i], neigh_x, neigh_y]
    #         weight_val = weights[0, act_chinds[act_i], sense_chinds[sense_i]]
    #         mem[0, act_chinds[act_i], center_x, center_y] += weight_val * sensor_val

    # @ti.kernel
    # def add_bias(self, mem: ti.types.ndarray(), bias: ti.types.ndarray(),
    #              act_chinds: ti.types.ndarray(), cell_coords: ti.types.ndarray()):
    #     for cell_i, act_i in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
    #         center_x = cell_coords[cell_i, 0]
    #         center_y = cell_coords[cell_i, 1]
    #         mem[0, act_chinds[act_i], center_x, center_y] += bias[0, act_chinds[act_i], 0]

    
    # @ti.kernel
    # def mm_weights(self, mem: ti.types.ndarray(), weights: ti.types.ndarray(), biases: ti.types.ndarray(),
    #              cell_coords: ti.types.ndarray(), kernel: ti.types.ndarray(),
    #              sense_chinds: ti.types.ndarray(), act_chinds: ti.types.ndarray()):
    #     for c, k in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
    #         i = cell_coords[c,0]
    #         j = cell_coords[c,1]
    #         mem[0,k,i,j] = mem[0,k,(i+1)%mem.shape[2],j]

        # for cell_i, act_j in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
        #     center_x = cell_coords[cell_i, 0]
        #     center_y = cell_coords[cell_i, 1]
        #     center_val = mem[0, act_chinds[act_j], center_x, center_y]
        #     left_val = mem[0, act_chinds[act_j], (center_x-1)%mem.shape[2], center_y]
        #     # for off_x, off_y in ti.ndrange((-1,2), (-1,2)):
        #     #     nx = (center_x + off_x) % mem.shape[2]
        #     #     ny = (center_y + off_y) % mem.shape[3]
        #     #     act_val += mem[0, act_chinds[act_j], nx, ny]
        #     mem[0, act_chinds[act_j], center_x, center_y] = left_val

        # for cell_i, act_j in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
        #     center_x = cell_coords[cell_i, 0]
        #     center_y = cell_coords[cell_i, 1]
        #     act_val = 0.0
        #     for off_n in ti.ndrange(kernel.shape[0]):
        #         neigh_x = (center_x + kernel[off_n, 0]) % mem.shape[2]
        #         neigh_y = (center_y + kernel[off_n, 1]) % mem.shape[3]
        #         for sense_n in ti.ndrange(sense_chinds.shape[0]):
        #             sensor_val = mem[0, sense_chinds[sense_n], neigh_x, neigh_y]
        #             weight_val = weights[0, act_j, sense_n]
        #             act_val += sensor_val#weight_val * sensor_val
        #     mem[0, act_chinds[act_j], center_x, center_y] = act_val #+ biases[0, act_chinds[act_j], 0]

    @ti.kernel
    def mm_2(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(), cell_coords: ti.types.ndarray(),
             kernel: ti.types.ndarray(), sense_chinds: ti.types.ndarray(), act_chinds: ti.types.ndarray(),
             weights: ti.types.ndarray(), biases: ti.types.ndarray()):
        for cell_i, act_j in ti.ndrange(cell_coords.shape[0], act_chinds.shape[0]):
            val = 0.0
            center_x = cell_coords[cell_i, 0]
            center_y = cell_coords[cell_i, 1]
            for sensor_n, off_m in ti.ndrange(sense_chinds.shape[0], kernel.shape[0]):
                neigh_x = (center_x + kernel[off_m, 0]) % mem.shape[2]
                neigh_y = (center_y + kernel[off_m, 1]) % mem.shape[3]
                val += mem[0, sensor_n, neigh_x, neigh_y] * weights[0, act_j, sensor_n]
            out_mem[0, act_j, center_x, center_y] = val + biases[0, act_j, 0]

    def forward(self, mem, genome_map):
        with torch.no_grad():
            # mem += torch.randn_like(mem) * 0.1
            cell_coords = self.get_cell_coords(genome_map)
            out_mem = torch.zeros_like(mem)
            self.mm_2(mem, out_mem, cell_coords,
                      self.kernel, self.sense_chinds, self.act_chinds,
                      self.net.weights, self.net.biases)
            # sensor_mem = torch.zeros(cell_coords.shape[0],
            #                          self.kernel.shape[0] * self.n_senses,
            #                          dtype=torch.float32, device=self.torch_device)
            # self.sense_to(mem, sensor_mem, cell_coords, self.sense_chinds, self.kernel)
            # actions = self.net.activate(sensor_mem)
            # self.store_actions(actions, mem, self.act_chinds, cell_coords)
            # self.add_bias(mem, self.net.out_bias, self.act_chinds, cell_coords)
            mem = nn.ReLU()(out_mem)
            # Calculate the mean across batch and channel dimensions
            mean = mem.mean(dim=(0, 2, 3), keepdim=True)
            # Calculate the variance across batch and channel dimensions
            var = mem.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            # Normalize the input tensor
            mem.sub_(mean).div_(torch.sqrt(var + 1e-5))

            # mem = ch_norm(mem)
            mem = torch.sigmoid(mem)
            self.substrate.mem = mem
            # mem[:, self.act_chinds] = ch_norm(mem[:, self.act_chinds])
            # mem[:, self.act_chinds] = torch.sigmoid(mem[:, self.act_chinds])
            # return mem
