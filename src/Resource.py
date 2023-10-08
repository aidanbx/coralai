import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.stats import uniform
from scipy.stats import levy_stable
import importlib
import src.pcg as pcg
importlib.reload(pcg)

class Resource:
    # a resource as an id (incremental), a min and max,
    # regeneration and dispersal functions of time,
    # a distribution function (levy dust usually)

    # a resource exists in the resource map as an id on a given cell along with a value
    # the id determines how to update it using the respective functions in this class

    def __init__(self, id, regen_func, metadata=None, dispersal_func=None):
        self.id = id
        self.regen_func = regen_func
        default_metadata = {
            'id': id
            }
        if metadata is None:
            metadata = {}
        metadata.update(default_metadata)
        self.metadata = metadata
        self.dispersal_func = dispersal_func