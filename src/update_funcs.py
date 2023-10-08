import numpy as np
import torch
import importlib
import src.Simulation as Simulation
importlib.reload(Simulation)
import src.Channel as Channel
importlib.reload(Channel)

def regen_ports(sim: Simulation, ports: Channel):
    # TODO: take into account resource size and obstacles?
    period = sim.metadata["period"]
    port_id_map = ports.metadata["port_id_map"]
    port_sizes = ports.metadata["port_sizes"]
    resources = ports.metadata["resources"]
    for resource in resources:
        ports.contents[port_id_map == resource.id] += resource.regen_func(period)
        torch.clamp(ports.contents[port_id_map == resource.id],
                    ports.allowed_range[0], ports.allowed_range[1],
                    out=ports.contents[port_id_map == resource.id])
    num_regens = ports.metadata.get("num_regens", 0)
    ports.metadata.update({"num_regens": num_regens + 1})
    return ports

# def grow(self, actuators, muscle_radii, capital, growth_cost):
#     muscle_radii_delta = actuators[self.actuator_indecies["muscle_radii_delta"]]
#     physics.grow_muscle_csa(self.cfg, muscle_radii, muscle_radii_delta, capital, growth_cost)

# def flow(self, actuators, capital, waste, muscle_radii, flow_cost, obstacles):
#     flow_act = actuators[self.actuator_indecies["flow_act"]]
#     physics.activate_flow_muscles(self.cfg, capital, waste, muscle_radii[:-2], flow_act, flow_cost, obstacles)

# def eat(self, actuators, capital, port, muscle_radii, port_cost):
#     port_act = actuators[self.actuator_indecies["port_act"]]
#     physics.activate_port_muscles(self.cfg, capital, port, muscle_radii[-2], port_act, port_cost)

# def dig(self, actuators, muscle_radii, mine_act, capital, obstacles, waste, mining_cost):
#     mine_act = actuators[self.actuator_indecies["mine_act"]]
#     physics.activate_mine_muscles(muscle_radii[-1], mine_act, capital, obstacles, waste, mining_cost)



