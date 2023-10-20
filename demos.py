import json
from matplotlib import gridspec, pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import torch
import numpy as np
from src import pcg, physics
from src.Simulation import Simulation
import src.EINCASM as EINCASM
import src.visualizers as visualizers

def random_arrow_demo():
    # HEX
    kernel = torch.tensor([
        [0, 0],     # ORIGIN
        [-1, 0],    # N
        [0, 1.0],   # W
        [1, 0],     # S
        [0, -1]     # E
    ], dtype=torch.int8)

    sim = Simulation(world_shape=(10,10))

    sim.add_channel('capital')
    sim.add_channel('muscles', num_layers=kernel.shape[0], metadata={'kernel': kernel})
    sim.add_channel('deltas', num_layers=kernel.shape[0])

    def add_random(sim, cap, musc, deltas, md):
        musc.contents += torch.rand(musc.contents.shape)*0.1 + 0.01

    sim.add_rule('grow', add_random,
                input_channel_ids=['capital','muscles','deltas'],
                affected_channel_ids=['muscles', 'capital'],
                metadata={'growth_cost': 0.1,})

    sim.init_all_channels()
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    def update(frame):
        sim.apply_rule('grow')
        for x in range(sim.world_shape[0]):
            for y in range(sim.world_shape[1]):
                for k in range(kernel.shape[0]):
                    dx, dy = kernel[k]
                    magnitude = sim.channels['muscles'].contents[k, x, y]
                    ax_br.arrow(x, y, dx * magnitude, dy * magnitude*0.05, head_width=0.1, head_length=0.2, fc='k', ec='k')

    ani = FuncAnimation(fig, update, frames=100, interval=50, repeat=True)
    plt.tight_layout()
    plt.show()

# def resource_weather_demo():
#     sim = Simulation("Resource Weather Demo")
#     sim.add_channel("ports", init_func=pcg.init_ports_levy, allowed_range=[-1,10],
#                     metadata={'description': 'Currently +/- resources generated with levy dust',
#                             'num_resources': 3,
#                             'min_regen_amp': 0.5,
#                             'max_regen_amp': 2,
#                             'alpha_range': [0.4, 0.9],
#                             'beta_range': [0.8, 1.2],
#                             'num_sites_range': [50, 100]})
    
#     sim.metadata.update({'period': 0.0})
#     sim.add_rule("step_period",
#                             lambda sim: sim.metadata.update({"period": sim.metadata["period"] + np.pi/100}),
#                             input_channel_ids=[], affected_channel_ids=[],
#                             metadata={'description': 'Increment period'},
#                             req_sim_metadata = {"period": float})

#     sim.add_rule("regen_resources",
#                                 physics.regen_ports,
#                                 input_channel_ids=["ports"], affected_channel_ids=["ports"],
#                                 metadata={'description': 'Regenerate resources'},
#                                 req_channel_metadata = {"ports": ["port_id_map", "port_sizes", "resources"]},
#                                 req_sim_metadata = {"period": float})

#     sim.init_all_channels()
#     resources = sim.channels["ports"].metadata["resources"]
#     port_id_map = sim.channels["ports"].metadata["port_id_map"]
#     ports = sim.channels["ports"].contents.squeeze(0)
#     allowed_range = sim.channels["ports"].allowed_range
#     # periods = torch.linspace(0, np.pi*8, 1s000)

#     fig = plt.figure(figsize=(12, 8))
#     gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.25])
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[1, :])

#     resource_colors = ['red', 'green', 'blue']
#     cmap = mcolors.ListedColormap(['white'] + resource_colors)
#     midmap = mcolors.TwoSlopeNorm(vmin=allowed_range[0], vcenter=0, vmax=allowed_range[1])

#     def update(frame_number):
#         period = sim.metadata["period"]

#         fig.suptitle(f"Period: {period:.2f}")
#         # Top-left: Resource ID map
#         ax1.clear()
#         ax1.set_title("Port ID map")
#         ax1.imshow(port_id_map, cmap=cmap)

#         # Top-right: Resource values
#         ax2.clear()
#         ax2.set_title("Port Values (Yl Good)")
#         ax2.imshow(ports, cmap="viridis", norm=midmap)
                        
#         ax3.clear()
#         ax3.set_title("Regeneration functions")
#         for idx, resource in enumerate(resources):
#             t = torch.linspace(0, 2*np.pi, 100)
#             y = resource.regen_func(t+period-np.pi/2).numpy()
#             ax3.plot(t+period-np.pi/2, y, color=resource_colors[idx], label=f"Resource {idx+1}")

#         ax3.axvline(x=period, color='black', linestyle='--')  # Static vertical line
#         ax3.axvspan(period, period-np.pi/2, facecolor='gray', alpha=0.2)  # Highlight indicating past
#         ax3.legend(loc="upper right")
        
#         sim.apply_all_rules()

#     ani = FuncAnimation(fig, update, frames=100, interval=50, repeat=True)
#     plt.tight_layout()
#     plt.show()


def EINCASM_demo():
    tst = EINCASM.EINCASM((10,10), verbose=True)
    # a = json.dumps(tst.sim.channels["capital"].metadata)
    tst.sim.init_all_channels()

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,0.25])
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tm = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[0, 2])
    ax_ml = fig.add_subplot(gs[1, 0])
    ax_mm = fig.add_subplot(gs[1, 1])
    ax_mr = fig.add_subplot(gs[1, 2])
    ax_bl = fig.add_subplot(gs[2, 0])
    ax_b = fig.add_subplot(gs[2, 1:]) 

    ax_port = ax_ml
    ax_rmap = ax_tl
    ax_capital = ax_mm
    ax_activation = ax_ml
    ax_obstacle = ax_ml
    ax_waste = ax_tl
    ax_activation = ax_tm
    ax_obstacle = ax_tr

    ax_regen = ax_b

    def update(frame):
        tst.sim.apply_all_rules()
        visualizers.vis_weather(tst.sim, ax_port, ax_rmap, ax_regen, tst.sim.channels[EINCASM.PORTS], {})

        ax_capital.clear()
        ax_capital.set_title("Capital")
        ax_capital.imshow(tst.sim.channels[EINCASM.CAPITAL].contents.squeeze(0), cmap="viridis", norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=2, vmax=4))

        ax_obstacle.clear()
        ax_obstacle.set_title("Obstacles")
        ax_obstacle.imshow(tst.sim.channels[EINCASM.OBSTACLES].contents.squeeze(0), cmap="gray", norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))

        ax_waste.clear()
        ax_waste.set_title("Waste")
        ax_waste.imshow(tst.sim.channels[EINCASM.WASTE].contents.squeeze(0), cmap="viridis", norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=2, vmax=4))

        ax_activation.clear()
        ax_activation.set_title("Flow Activation")
        print(f"flow_mact: {tst.sim.channels[EINCASM.FLOW_MACT].contents.sum()}")
        print(f"mact: {tst.sim.channels[EINCASM.ALL_MUSCLE_ACT].contents.sum()}")
        ax_activation.imshow(tst.sim.channels[EINCASM.FLOW_MACT].contents.squeeze(0), cmap="viridis", norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))

    ani = FuncAnimation(fig, update, frames=100, interval=50, repeat=True)
    plt.tight_layout()
    plt.show()
    # for _ in range(10):
    #     print(repr(tst.sim.channels[EINCASM.MUSCLES]))
    #     try:
    #         tst.sim.iterate()
    #     except Exception as e:
    #         raise RuntimeError(f"{repr(tst.sim.channels[EINCASM.MUSCLES])}\n\n{e}") from e

    #     print('----------------------------------')

    # print(repr(tst.sim.channels[EINCASM.MUSCLES]))



if __name__ == "__main__":
    # resource_weather_demo()
    # random_arrow_demo()
    EINCASM_demo()