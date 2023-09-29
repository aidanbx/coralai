import os
import sys

from ..src.generate_env import *
# import src.generate_env as genv
# import src.eincasm_config as eincasm_config
# import src.visualize as visualize
import tester

# TEST: 0
# ---------------
config = eincasm_config.Config('./config.yaml')


verbose = True
tester.test(lambda: genv.generate_env(config, visualize=True),
            "Generate Environment",
            verbose,
            lambda result, title: visualize.show_image(result[1]))
# ---------------