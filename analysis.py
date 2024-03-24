import os
import json
import torch
import taichi as ti

from coralai.substrate.visualization import Visualization
from coralai.substrate.substrate import Substrate

class AnalysisVis(Visualization):
    def __init__(self, substrate, chids, run_name):
        super().__init__(substrate, chids)
        self.run_name = run_name
        self.current_step_index = 0  # Initialize current step index to 0
        self.steps = []  # This will hold the actual step numbers extracted from folder names

    def render_opt_window(self):
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(600 / self.img_w, self.img_w)
        opt_h = min(200 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            # UI for changing RGB channels
            self.chinds[0] = sub_w.slider_int(
                f"R: {self.substrate.index_to_chname(self.chinds[0])}", 
                self.chinds[0], 0, self.substrate.mem.shape[1]-1)
            self.chinds[1] = sub_w.slider_int(
                f"G: {self.substrate.index_to_chname(self.chinds[1])}", 
                self.chinds[1], 0, self.substrate.mem.shape[1]-1)
            self.chinds[2] = sub_w.slider_int(
                f"B: {self.substrate.index_to_chname(self.chinds[2])}", 
                self.chinds[2], 0, self.substrate.mem.shape[1]-1)
            # Display the experiment run name
            sub_w.text(f"Experiment Run: {self.run_name}")
            # Display the actual step number
            if self.steps:
                sub_w.text(f"Step Number: {self.steps[self.current_step_index]}")

    def update(self, current_step_index, steps):
        self.current_step_index = current_step_index  # Update the current step index
        self.steps = steps  # Update the list of steps with actual step numbers
        super().update()  # Call the update method of the base class


def display_steps(run_dir, substrate):
    ti.init(ti.metal)
    steps = sorted([d for d in os.listdir(run_dir) if d.startswith("step_")],
                   key=lambda x: int(x.split("_")[1]))
    current_step_index = 0

    substrate.mem = load_sub_mem(os.path.join(run_dir, steps[current_step_index]))
    
    run_name = os.path.basename(run_dir)  # Extract the run name from the run directory path
    vis = AnalysisVis(substrate, chids=["energy", "infra", "genome"], run_name=run_name)  # Adjust chids as needed

    escaped = False
    while vis.window.running and not escaped:
        for e in vis.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.RIGHT:
                current_step_index = min(current_step_index + 1, len(steps) - 1)
                substrate.mem = load_sub_mem(os.path.join(run_dir, steps[current_step_index]))
            elif e.key == ti.ui.LEFT:
                current_step_index = max(current_step_index - 1, 0)
                substrate.mem = load_sub_mem(os.path.join(run_dir, steps[current_step_index]))
            elif e.key == ti.ui.ESCAPE:
                escaped = True

        vis.update(current_step_index, steps)

class Run:
    def __init__(self, data_path, torch_device):
        self.data_path = data_path
        self.torch_device = torch_device
        self.substrate = Run.gen_substrate(self.data_path, self.torch_device)
    

    def construct_channel_dtype(channel_data):
        # If the channel has subchannels, construct a struct
        if "subchannels" in channel_data:
            subchannel_types = {subchannel: Run.construct_channel_dtype(subchannel_data)
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
            channels[channel_name] = Run.construct_channel_dtype(channel_data)

        # Initialize the substrate with the extracted shape and channels
        substrate = Substrate(shape, torch.float32, torch_device, channels)
        substrate.malloc()  # Allocate memory for the substrate

        return substrate

    def load_sub_mem(step_dir):
        sub_mem_path = os.path.join(step_dir, "sub_mem")
        sub_mem = torch.load(sub_mem_path)
        return sub_mem
    

    
class History:
    def __init__(self, history_path, torch_device):
        self.history_path = history_path
        self.torch_device = torch_device
        self.runs = os.listdir(self.history_path)
        self.current_run_num = 0
        self.current_run_name = self.runs[self.current_run_num]
        self.init_curr_run()


    def goto(self, run_name=None, run_num=None):
        if not run_name and not run_num:
            raise ValueError("Must provide either run_name or run_num")
        if run_name and run_num:
            raise ValueError("Must provide either run_name or run_num, not both")
        if run_name:
            self.current_run_num = self.runs.index(run_name)
            self.current_run_name = run_name
        if run_num:
            self.current_run_num = run_num
            self.current_run_name = self.runs[run_num]
        self.init_curr_run()


    def next_run(self):
        self.current_run_num = (self.current_run_num + 1) % len(self.runs)
        self.current_run_name = self.runs[self.current_run_num]
        self.init_curr_run()
    

    def prev_run(self):
        self.current_run_num = (self.current_run_num - 1) % len(self.runs)
        self.current_run_name = self.runs[self.current_run_num]
        self.init_curr_run()


    def init_curr_run(self):
        self.current_run = Run(os.path.join(self.history_path, self.current_run_name), self.torch_device)


def create_vis(run_dirs, run_num, torch_device):
    substrate = gen_substrate(run_dir, torch_device)
    display_steps(run_dir, substrate)


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    run_dirs = os.listdir("./history/")
    hist_dir = "./history/"

    run_name = "space_evolver_run_240310-0013_40"
    run_dir = os.path.join(hist_dir, run_name)

    create_vis(run_dir, torch_device)
    run_name = "space_evolver_run_240309-0248_08"
    run_dir = os.path.join(hist_dir, run_name)

    create_vis(run_dir, torch_device)
