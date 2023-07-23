
# %% 

import colorsys
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2  # You might need OpenCV for HSV conversions
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


def collate_channel_images(config, images):
    super_image = np.zeros((images.shape[1], images.shape[2], 4))

    for img in images:
        # Mask for the current channel where its value is greater than 0
       super_image[img[..., 3] > 0] = img[img[..., 3] > 0]

    return super_image
    

def channels_to_images(config, channels, colormaps=None):
    if colormaps is None:
        # food, poison, obstacle
        colormaps = config["visualize"]["colormaps"]
    assert len(colormaps) == len(channels)

    images=[]
    for channel_name in config["environment"]["channels"]:
        index = config["environment"]["channels"].index(channel_name)
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

def blend_colors(color1, color2, weight):
    # Convert RGB colors to HSV
    hsv1 = colorsys.rgb_to_hsv(*color1)
    hsv2 = colorsys.rgb_to_hsv(*color2)

    # Blend the colors based on the weight
    blended_hsv = (
        (1 - weight) * hsv1[0] + weight * hsv2[0],    # Weighted average of the hues
        (1 - weight) * hsv1[1] + weight * hsv2[1],    # Weighted average of the saturations
        (1 - weight) * hsv1[2] + weight * hsv2[2]     # Weighted average of the values/brightness
    )

    # Convert the blended color back to RGB
    blended_rgb = colorsys.hsv_to_rgb(*blended_hsv)

    return blended_rgb


# %%

# config = load_check_config("config.yaml")
# channels = np.load("channels.npy")
