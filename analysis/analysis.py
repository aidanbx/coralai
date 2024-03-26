from dataclasses import dataclass
import os
from typing import List
from matplotlib import pyplot as plt
import torch
import taichi as ti

from analysis_vis_mpl import visualize_run_data
from analysis_data import HistoryData

if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    # run_dirs = os.listdir("./history/")
    hist_dir = "./history/"

    run_name = "space_evolver_run_240309-0308_36"
    hist_data = HistoryData(hist_dir, torch_device)
    hist_data.goto_run_name(run_name)

    visualize_run_data(hist_data)
