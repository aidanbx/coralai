from generate_env import *

import yaml
import visualize
import tester

# TEST: 0
# ---------------
with open("./config.yaml", "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

verbose = True
tester.test(lambda: generate_env(config, visualize=True),
            "Generate Environment",
            verbose,
            lambda result, title: visualize.show_image(result[1]))
# ---------------