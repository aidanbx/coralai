from dataclasses import dataclass
import os
import json
import pickle
from typing import List
import torch
import taichi as ti
import neat

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

def fetch_genomes_at_step(step_dir):
    # genomes is a pickle file
    genome_path = os.path.join(step_dir, "genomes")
    with open(genome_path, "rb") as f:
        genomes = pickle.load(f)
    return genomes


@dataclass
class RunData:
    run_path: str
    run_name: str
    substrate: Substrate
    genomes: List[neat.DefaultGenome]
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
        self.genomes = fetch_genomes_at_step(self.step_path)

    def next_step(self):
        self.curr_step_index += 1
        self.load_step()

    def prev_step(self):
        self.curr_step_index -= 1
        self.load_step()


@dataclass
class HistoryData:
    hist_path: str
    run_names: List[str]
    curr_run_index: int
    curr_run_path: str
    curr_run_data: RunData
    torch_device: torch.DeviceObjType

    def __init__(self, hist_path, torch_device):
        self.hist_path = hist_path
        self.run_names = [run_name for run_name in os.listdir(hist_path)]
        self.curr_run_index = 0
        self.curr_run_path = None
        self.curr_run_data = None
        self.torch_device = torch_device

        self.load_run()

    def load_run(self):
        self.curr_run_index = self.curr_run_index % len(self.run_names)
        self.curr_run_path = os.path.join(self.hist_path, self.run_names[self.curr_run_index])
        self.curr_run_data = RunData(self.curr_run_path, self.torch_device)

    def next_run(self):
        self.curr_run_index += 1
        self.load_run()

    def prev_run(self):
        self.curr_run_index -= 1
        self.load_run()

    def goto_run_name(self, run_name):
        self.curr_run_index = self.run_names.index(run_name)
        self.load_run()
