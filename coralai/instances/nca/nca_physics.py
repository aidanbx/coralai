from ...dynamics.Physics import Physics

class NCAPhysics(Physics):
    def __init__(self):
        super(NCAPhysics, self).__init__()
        pass

    def apply_actuators(self, world, actuator_values):
        world.mem = actuator_values