import os
import json
import torch
from dataclasses import dataclass
from coralai.substrate.substrate import Substrate
from coralai.substrate.visualization import VisualizationData, add_channel_controls, compose_visualization

@dataclass
class AnalysisVisData(VisualizationData):
    run_dir: str
    run_name: str

    def __init__(self, substrate: Substrate, chids: list = None, window_w=800, name="Analysis Visualization"):
        super().__init__(substrate, chids, window_w, name)
        self.run_dir = None
        self.run_name = None

    def set_run_dir(self, run_dir):
        self.run_dir = run_dir
        self.run_name = os.path.basename(run_dir)


def add_analysis_controls(analysis_vis_data: AnalysisVisData, sub_window):
    sub_window.text(f"Experiment Run: {analysis_vis_data.run_name}")
    return analysis_vis_data


def compose_analysis_vis(analysis_vis_data: AnalysisVisData):
    return compose_visualization(analysis_vis_data, [add_channel_controls, add_analysis_controls], [])
