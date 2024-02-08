class Organism:
    def __init__(self, world, sensors, n_actuators):
        self.world = world
        self.w = world.w
        self.h = world.h
        self.sensors = sensors
        self.sensor_inds = self.world.windex[self.sensors]
        self.n_sensors = self.sensor_inds.shape[0]
        self.n_actuators = n_actuators

    def forward(self, x=None):
        return x

    def perturb_weights(self, perturbation_strength):
        pass