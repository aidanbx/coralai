class Organism:
    def __init__(self, n_sensors, n_actuators):
        self.n_sensors = n_sensors
        self.n_actuators = n_actuators

    def forward(self, x):
        return x
    
    def mutate(self):
        return self