# Testing connecting generate_env to HexagonGrid

import generate_env
import numpy as np
from Hexagon import *
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

env = generate_env.generate_env(config, visualize=False)
print("yo")




