import json
from matplotlib import gridspec, pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import torch
import numpy as np
from src import pcg, physics
from src.Simulation import Simulation
import src.EINCASM as EINCASM

def resource_weather_demo():
    sim = Simulation("Resource Weather Demo")
    sim.add_channel("ports", init_func=pcg.init_ports_levy, allowed_range=[-1,10],
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
                                physics.regen_ports,
                                input_channel_ids=["ports"], affected_channel_ids=["ports"],
                                metadata={'description': 'Regenerate resources'},
                                req_channel_metadata = {"ports": ["port_id_map", "port_sizes", "resources"]},
                                req_sim_metadata = {"period": float})

    sim.init_all_channels()
    resources = sim.channels["ports"].metadata["resources"]
    port_id_map = sim.channels["ports"].metadata["port_id_map"]
    ports = sim.channels["ports"].contents.squeeze(0)
    allowed_range = sim.channels["ports"].allowed_range
    # periods = torch.linspace(0, np.pi*8, 1s000)

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


def torch_NCA():
    import torch
    import torch.nn as nn

    class NCAModel(nn.Module):
        def __init__(self, channel_count):
            super(NCAModel, self).__init__()
            state = torch.rand(2, n_channels, 100, 100)
            state[:,3,...]=torch.zeros_like(state[:,3,...])
            state[:,3,45:55,45:55]=torch.ones_like(state[:,3,45:55,45:55])
            self.state = state
            self.conv = nn.Conv2d(channel_count, channel_count, kernel_size=3, padding=1, padding_mode='circular')

        def forward(self, x):
            # Apply the convolutional layer
            x = self.conv(x)
            
            # Apply the activation function
            x = nn.ReLU()(x)
            # batch norm
            x = nn.BatchNorm2d(x.shape[1])(x)
            # Ensure output is within 0..1 for image data
            x = torch.sigmoid(x)

            return x

    def extract_ims(state):
        return state.detach().permute(0,2,3,1)[:,:,:,:4].numpy()
    
    # Define the update function for the animation
    def update(f, model):
        state = model(model.state)
        model.state = state
        img1, img2 = extract_ims(state)
        im1.set_array(img1)
        im2.set_array(img2)

        # add some random noise to state
        state += torch.rand_like(state)*0.1-0.05
    
        return im1, im2

    n_hidden = 7
    n_channels = 4 + n_hidden
    # Initialize the state
    
    # Create the model
    model = NCAModel(channel_count=n_channels)

    # Create the figure for the animation
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    ims = extract_ims(model.state)

    im1 = axs[0].imshow(ims[0])
    im2 = axs[1].imshow(ims[1])

    # Create the animation
    ani = FuncAnimation(fig, lambda frame: update(frame, model), frames=100, interval=50, repeat=True)

    # Display the animation
    plt.show()

if __name__ == "__main__":
    # torch_NCA()
    # resource_weather_emdo()
    tst = EINCASM.EINCASM((2,2), verbose=True)
    # a = json.dumps(tst.sim.channels["capital"].metadata)
    tst.sim.init_all_channels()
    for _ in range(10):
        print(repr(tst.sim.channels[EINCASM.MUSCLES]))
        try:
            tst.sim.update()
        except Exception as e:
            raise RuntimeError(f"{repr(tst.sim.channels[EINCASM.MUSCLES])}\n\n{e}") from e

        print('----------------------------------')

    # print(repr(tst.sim.channels[EINCASM.MUSCLES]))
