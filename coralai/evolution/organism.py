class Organism:
    def __init__(self, substrate, sensors, n_actuators):
        self.substrate = substrate
        self.sensors = sensors
        self.sensor_inds = substrate.windex[sensors]
        self.n_sensors = len(self.sensor_inds)
        self.n_actuators = n_actuators

    def forward(self, x):
        return x
    
    def mutate(self):
        return self