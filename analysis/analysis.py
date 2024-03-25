import os
import json
import torch
import taichi as ti

from analysis_vis import compose_analysis_vis, AnalysisVisData
from coralai.substrate.substrate import Substrate


# def display_steps(run_dir, substrate):
#     ti.init(ti.metal)
#     steps = sorted([d for d in os.listdir(run_dir) if d.startswith("step_")],
#                    key=lambda x: int(x.split("_")[1]))
#     current_step_index = 0

#     substrate.mem = load_sub_mem(os.path.join(run_dir, steps[current_step_index]))
    
#     run_name = os.path.basename(run_dir)  # Extract the run name from the run directory path
#     vis = AnalysisVis(substrate, chids=["energy", "infra", "genome"], run_name=run_name)  # Adjust chids as needed

#     escaped = False
#     while vis.window.running and not escaped:
#         for e in vis.window.get_events(ti.ui.PRESS):
#             if e.key == ti.ui.RIGHT:
#                 current_step_index = min(current_step_index + 1, len(steps) - 1)
#                 substrate.mem = load_sub_mem(os.path.join(run_dir, steps[current_step_index]))
#             elif e.key == ti.ui.LEFT:
#                 current_step_index = max(current_step_index - 1, 0)
#                 substrate.mem = load_sub_mem(os.path.join(run_dir, steps[current_step_index]))
#             elif e.key == ti.ui.ESCAPE:
#                 escaped = True

#         vis.update(current_step_index, steps)

# class Run:
#     def __init__(self, data_path, torch_device):
#         self.data_path = data_path
#         self.torch_device = torch_device
#         self.substrate = Run.gen_substrate(self.data_path, self.torch_device)
    

def construct_channel_dtype(channel_data):
    # If the channel has subchannels, construct a struct
    if "subchannels" in channel_data:
        subchannel_types = {subchannel: construct_channel_dtype(subchannel_data)
                            for subchannel, subchannel_data in channel_data["subchannels"].items()}
        return ti.types.struct(**subchannel_types)
    else:
        # Assuming all channels without subchannels are of type float32
        return ti.f32
    

def gen_substrate(experiment_dir, torch_device):
    sub_metadata_path = os.path.join(experiment_dir, "sub_meta")
    sub_meta = None
    with open(sub_metadata_path, "r") as f:
        sub_meta = json.load(f)

    # Extract shape from metadata
    shape = sub_meta["shape"][2:]  # Assuming the relevant dimensions are the last two

    # Construct channels dictionary with proper handling of subchannels
    channels = {}
    for channel_name, channel_data in sub_meta["windex"].items():
        channels[channel_name] = construct_channel_dtype(channel_data)

    # Initialize the substrate with the extracted shape and channels
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()  # Allocate memory for the substrate

    return substrate

def load_sub_mem(step_dir):
    sub_mem_path = os.path.join(step_dir, "sub_mem")
    sub_mem = torch.load(sub_mem_path)
    return sub_mem
    

    
# class History:
#     def __init__(self, history_path, torch_device):
#         self.history_path = history_path
#         self.torch_device = torch_device
#         self.runs = os.listdir(self.history_path)
#         self.current_run_num = 0
#         self.current_run_name = self.runs[self.current_run_num]
#         self.init_curr_run()


#     def goto(self, run_name=None, run_num=None):
#         if not run_name and not run_num:
#             raise ValueError("Must provide either run_name or run_num")
#         if run_name and run_num:
#             raise ValueError("Must provide either run_name or run_num, not both")
#         if run_name:
#             self.current_run_num = self.runs.index(run_name)
#             self.current_run_name = run_name
#         if run_num:
#             self.current_run_num = run_num
#             self.current_run_name = self.runs[run_num]
#         self.init_curr_run()


#     def next_run(self):
#         self.current_run_num = (self.current_run_num + 1) % len(self.runs)
#         self.current_run_name = self.runs[self.current_run_num]
#         self.init_curr_run()
    

#     def prev_run(self):
#         self.current_run_num = (self.current_run_num - 1) % len(self.runs)
#         self.current_run_name = self.runs[self.current_run_num]
#         self.init_curr_run()


#     def init_curr_run(self):
#         self.current_run = Run(os.path.join(self.history_path, self.current_run_name), self.torch_device)


# def create_vis(run_dirs, run_num, torch_device):
#     substrate = gen_substrate(run_path, torch_device)
#     display_steps(run_path, substrate)


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    # run_dirs = os.listdir("./history/")
    hist_dir = "./history/"

    run_name = "space_evolver_run_240310-0013_40"
    run_path = os.path.join(hist_dir, run_name)
    substrate = gen_substrate(run_path, torch_device)
    substrate.mem = load_sub_mem(os.path.join(run_path, "step_0"))

    avis_data = AnalysisVisData(substrate)
    avis_data.set_run_dir(run_path)
    update_analysis_vis = compose_analysis_vis(avis_data)

    while avis_data.window.running and not avis_data.escaped:
        update_analysis_vis()

    # create_vis(run_dir, torch_device)
    # run_name = "space_evolver_run_240309-0248_08"
    # run_dir = os.path.join(hist_dir, run_name)

    # create_vis(run_dir, torch_device)
