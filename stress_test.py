import taichi as ti
import torch

ti.init(ti.metal)
device=torch.device("mps")

w=1024
h=1024
n_senses=6
n_acts=10
kernel = torch.tensor(
    [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
    device=device
)

weights = torch.randn((w, h, n_senses*kernel.shape[0], n_acts), device=device)
biases = torch.randn((w, h, n_acts), device=device)

act_mem = torch.randn((n_acts, w, h), device=device)
substrate = torch.randn((n_senses, w, h), device=device)

@ti.kernel
def apply_weights_and_biases(in_mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                             weight_grid: ti.types.ndarray(), bias_grid: ti.types.ndarray(),
                             kernel: ti.types.ndarray()):
    for i, j, act_k in ti.ndrange(in_mem.shape[1], in_mem.shape[2], out_mem.shape[0]):
        val = 0.0
        for sense_ch_n in ti.ndrange(out_mem.shape[0]):
            for offset_m in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[offset_m, 0]) % out_mem.shape[1]
                neigh_y = (j + kernel[offset_m, 1]) % out_mem.shape[2]
                weight_ind = int((sense_ch_n * kernel.shape[0]) + offset_m)
                val += (in_mem[sense_ch_n, neigh_x, neigh_y] *
                        weight_grid[i, j, act_k, weight_ind])
                
        out_mem[act_k, i, j] = val + bias_grid[i, j, act_k]

import time

num_runs = 10
total_time = 0

for _ in range(num_runs):
    print('a')
    start_time = time.time()
    apply_weights_and_biases(substrate, act_mem, weights, biases, kernel)
    end_time = time.time()
    total_time += (end_time - start_time)

average_time_per_run = total_time / num_runs
applications_per_second = 1 / average_time_per_run
print(f"Average applications per second over {num_runs} runs: {applications_per_second}")
