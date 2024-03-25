from dataclasses import dataclass
import os
import json
from typing import List
import torch
import taichi as ti

from analysis_vis import compose_analysis_vis, AnalysisVisData
from coralai.substrate.substrate import Substrate


def construct_channel_dtype(channel_data):
    if "subchannels" in channel_data:
        subchannel_types = {}
        for subchannel, subchannel_data in channel_data["subchannels"].items():
            subchannel_type = construct_channel_dtype(subchannel_data)
            subchannel_types[subchannel] = subchannel_type
        channel_type = ti.types.struct(**subchannel_types)
    elif len(channel_data["indices"]) == 1:
        channel_type = ti.f32
    elif len(channel_data["indices"]) > 1:
        channel_type = ti.types.vector(len(channel_data["indices"]), ti.f32)
    return channel_type

def load_substrate_metadata(experiment_dir):
    sub_metadata_path = os.path.join(experiment_dir, "sub_meta")
    with open(sub_metadata_path, "r") as f:
        sub_meta = json.load(f)
    shape = sub_meta["shape"][2:]
    channels = {}
    for channel_name, channel_data in sub_meta["windex"].items():
        channel_type = construct_channel_dtype(channel_data)
        channels[channel_name] = channel_type
    return shape, channels

def fetch_substrate_step_mem(step_dir):
    sub_mem_path = os.path.join(step_dir, "sub_mem")
    sub_mem = torch.load(sub_mem_path)
    return sub_mem

@dataclass
class RunData:
    run_path: str
    run_name: str
    substrate: Substrate
    steps: List[str]
    curr_step_index: int
    curr_step_number: int
    step_path: str
    torch_device: torch.DeviceObjType

    def __init__(self, run_path, torch_device):
        shape, channels = load_substrate_metadata(run_path)
        self.substrate = Substrate(shape, torch.float32, torch_device, channels)
        self.substrate.malloc()
        self.steps = sorted([step for step in os.listdir(run_path) if step.startswith("step_")], key=lambda x: int(x.split("_")[1]))
        
        self.run_path=run_path
        self.run_name=os.path.basename(run_path)
        self.curr_step_index=0
        self.curr_step_number=None
        self.step_path=None
        self.torch_device=torch_device

        self.load_step()

    def load_step(self):
        self.curr_step_index = self.curr_step_index % len(self.steps)
        self.step_path = os.path.join(self.run_path, self.steps[self.curr_step_index])
        self.step_number = int(self.steps[self.curr_step_index].split("_")[1])
        self.substrate.mem = fetch_substrate_step_mem(self.step_path)

    def next_step(self):
        self.curr_step_index += 1
        self.load_step()

    def prev_step(self):
        self.curr_step_index -= 1
        self.load_step()


@dataclass
class History:
    path: str
    run_names: List[str]
    curr_run_index: int
    curr_run_data: RunData



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


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    # run_dirs = os.listdir("./history/")
    hist_dir = "./history/"

    run_name = "space_evolver_run_240310-0013_40"
    run_path = os.path.join(hist_dir, run_name)
    run_data = RunData(run_path, torch_device)
    avis_data = AnalysisVisData(run_data.substrate)
    avis_data.set_run_dir(run_path)
    update_analysis_vis = compose_analysis_vis(avis_data)

    while avis_data.window.running and not avis_data.escaped:
        if avis_data.next_step_clicked:
            run_data.next_step()
            avis_data.next_step_clicked = False
        if avis_data.prev_step_clicked:
            run_data.prev_step()
            avis_data.prev_step_clicked = False
        avis_data.run_name = run_data.run_name
        avis_data.step_number = run_data.step_number
        update_analysis_vis()

    # create_vis(run_dir, torch_device)
    # run_name = "space_evolver_run_240309-0248_08"
    # run_dir = os.path.join(hist_dir, run_name)

    # create_vis(run_dir, torch_device)
