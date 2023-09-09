from generate_env import *
import visualize

# TEST: 0
# ---------------
config = yaml.safe_load("./config.yaml")
verbose = True
tester.test(lambda: generate_env(config, visualize=True),
            "Generate Environment",
            verbose,
            lambda result, title: visualize.show_image(result[1]))
# ---------------