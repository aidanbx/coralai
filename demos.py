from matplotlib import gridspec, pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import torch
import numpy as np

import importlib
import src.Resource as Resource
importlib.reload(Resource)
import src.pcg as pcg
importlib.reload(pcg)
import src.Simulation as Simulation
importlib.reload(Simulation)
import src.channel_funcs as channel_funcs
importlib.reload(channel_funcs)
import src.update_funcs as update_funcs
importlib.reload(update_funcs)

def resource_weather_demo():
    sim = Simulation.Simulation("Resource Weather Demo")
    sim.add_channel("ports", init_func=channel_funcs.init_resources_levy, allowed_range=[-1,10],
                    metadata={'description': 'Currently +/- resources generated with levy dust',
                            'num_resources': 3,
                            'min_regen_amp': 0.5,
                            'max_regen_amp': 2,
                            'alpha_range': [0.4, 0.9],
                            'beta_range': [0.8, 1.2],
                            'num_sites_range': [50, 100]})
    
    sim.metadata.update({'period': 0.0})
    sim.add_update_function("step_period",
                            lambda sim: sim.metadata.update({"period": sim.metadata["period"] + np.pi/100}),
                            input_channel_ids=[], affected_channel_ids=[],
                            metadata={'description': 'Increment period'},
                            req_sim_metadata = {"period": float})

    sim.add_update_function("regen_resources",
                                update_funcs.regen_ports,
                                input_channel_ids=["ports"], affected_channel_ids=["ports"],
                                metadata={'description': 'Regenerate resources'},
                                req_channel_metadata = {"ports": ["port_id_map", "port_sizes", "resources"]},
                                req_sim_metadata = {"period": float})

    sim.init_all_channels()
    resources = sim.channels["ports"].metadata["resources"]
    port_id_map = sim.channels["ports"].metadata["port_id_map"]
    ports = sim.channels["ports"].contents
    allowed_range = sim.channels["ports"].allowed_range
    period = sim.metadata["period"]
    # periods = torch.linspace(0, np.pi*8, 1000)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.25])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    resource_colors = ['red', 'green', 'blue']
    cmap = mcolors.ListedColormap(['white'] + resource_colors)
    midmap = mcolors.TwoSlopeNorm(vmin=allowed_range[0], vcenter=0, vmax=allowed_range[1])

    def update(frame_number):
        period = sim.metadata["period"]

        fig.suptitle(f"Period: {period:.2f}")
        # Top-left: Resource ID map
        ax1.clear()
        ax1.set_title("Port ID map")
        ax1.imshow(port_id_map, cmap=cmap)

        # Top-right: Resource values
        ax2.clear()
        ax2.set_title("Port Values (Yl Good)")
        ax2.imshow(ports, cmap="viridis", norm=midmap)
                        
        ax3.clear()
        ax3.set_title("Regeneration functions")
        for idx, resource in enumerate(resources):
            t = torch.linspace(0, 2*np.pi, 100)
            y = resource.regen_func(t+period-np.pi/2).numpy()
            ax3.plot(t+period-np.pi/2, y, color=resource_colors[idx], label=f"Resource {idx+1}")

        ax3.axvline(x=period, color='black', linestyle='--')  # Static vertical line
        ax3.axvspan(period, period-np.pi/2, facecolor='gray', alpha=0.2)  # Highlight indicating past
        ax3.legend(loc="upper right")
        
        sim.update()

    ani = FuncAnimation(fig, update, frames=100, interval=50, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    resource_weather_demo()

    
