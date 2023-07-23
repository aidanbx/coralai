
# %% 

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml


"""
visualize:
  colormaps:
    food: "copper"
    poison: "summer"
    obstacle: "binary"
"""
def load_check_config(config_file):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        # asserts colormaps are of right length
        assert len(config["visualize"]["colormaps"]) == len(config["environment"]["channels"])
        # asserts colormaps are real matplotlib colormaps
        for cmap in config["visualize"]["colormaps"].values():
            assert cmap in plt.colormaps()

        return config


def collate_channel_images(config, channels, images):
    super_image = np.zeros((channels.shape[1], channels.shape[2], 4))
    for i, ch in enumerate(channels):
        super_image[ch > 0] = images[i][ch > 0]
    return super_image
    

def channels_to_images(config, channels, colormaps=None):
    if colormaps is None:
        # food, poison, obstacle
        colormaps = config["visualize"]["colormaps"]
    print(colormaps)
    assert len(colormaps) == len(channels)

    images=[]
    for channel_name in config["environment"]["channels"]:
        index = config["environment"]["channel_index"][channel_name]
        colormap = config["visualize"]["colormaps"][channel_name]
        images.append(channel_to_image(channels[index], colormap))
  
    return images


def channel_to_image(channel, cmap="gray"):
    norm = mpl.colors.Normalize(channel.min(), channel.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(channel)

# ["copper", "summer", "binary"] + ["gray"] * (channels.shape[0] - 3)

def show_image(image, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    return ax


# %%

# config = load_check_config("config.yaml")
# channels = np.load("channels.npy")
