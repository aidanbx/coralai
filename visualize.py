
# %% 

import colorsys
import traceback
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2  # You might need OpenCV for HSV conversions
import yaml

import importlib
import tester 
importlib.reload(tester)

# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    verbose = True # For testing
# %% --------------------------------------------------------------------------


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

# Test Blend Color
# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    # Test blend colors
    color1 = (1.0, 0, 0)
    color2 = (0, 1.0, 0)
    def show_blend_result(result, title):
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(np.full((10, 10, 3), color1))
        plt.axis('off')
        plt.title('Color 1')
        plt.subplot(1, 3, 2)
        plt.imshow(np.full((10, 10, 3), color2))
        plt.axis('off')
        plt.title('Color 2')
        plt.subplot(1, 3, 3)
        plt.imshow(np.full((10, 10, 3), result))
        plt.axis('off')
        plt.title('Blended')
        plt.show()
        print("\033[93mResult of test", title, ":", result, "\033[0m")

    tester.test(lambda: blend_colors(color1, color2, 0.5),
                "Blend colors",
                verbose,
                show_blend_result)
# %% --------------------------------------------------------------------------
    

def show_image(image, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    plt.show()
    return ax


def channel_to_image(channel, cmap="gray"):
    norm = mpl.colors.Normalize(channel.min(), channel.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(channel)


# %% Test Channel to Image
# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    channel = np.array([
        [0, 0, 0.25],
        [1, 0.5, 0],
        [0, 1, 0]
    ])
    tester.test(lambda: channel_to_image(channel, "copper"),
                "Channel to Image",
                verbose,
                lambda result, title: show_image(result))
# %% --------------------------------------------------------------------------


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

def show_images_result(result, title):
    for image in result:
        show_image(image)


# Test Channels to Images
# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "environment": {
            "channels": ["food", "poison", "obstacle"]
        },
        "visualize": {
            "colormaps": {
                "food": "Greens",
                "poison": "Oranges",
                "obstacle": "binary"
            }
        }
    }
    channels = np.array([
        [[0.2, 0.4, 0.6], [0.8, 0.7, 0]],   # food channel
        [[0, 0.7, 0.9], [0.1, 0.3, 0]],     # poison channel
        [[1, 1, 1], [0, 0, 0]]              # obstacle channel, binary
    ])

        # print("\033[93mResult of test", title, ":", result, "\033[0m")

    tester.test(lambda: channels_to_images(config, channels),
                "Channels to Images",
                verbose,
                show_images_result)
# %% --------------------------------------------------------------------------

# In the order provided, stacks images on top of each other (first is the background)
# Makes alpha 0 for overlapping pixels (so the top image is visible, taking precedence)
def stack_mask_images(images):
    super_image = np.zeros((images.shape[1], images.shape[2], 4))

    for img in images:
        # Mask for the current channel where its value is greater than 0
        mask = img[..., 3] > 0
        # Set the super image to the current image where the mask is true
        super_image[mask] = img[mask]

    return super_image


def collate_channel_images(config, images):
    super_image = np.zeros((images.shape[1], images.shape[2], 4))

    for img in images:
        # Mask for the current channel where its value is greater than 0
       super_image[img[..., 3] > 0] = img[img[..., 3] > 0]

    return super_image

# Test Collate Channel Images
# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "environment": {
            "channels": ["food", "poison", "obstacle"]
        },
        "visualize": {
            "colormaps": {
                "food": "Greens",
                "poison": "Oranges",
                "obstacle": "binary"
            }
        }
    }
    channels = np.array([
            [[0.2, 0, 0], [0.8, 1, 0]],   # food channel
            [[0, 0, 0], [0.1, 0, 1]],     # poison channel
            [[0, 1, 1], [0, 0, 0]]              # obstacle channel, binary
        ])
    images = channels_to_images(config, channels)
    show_images_result(images, "Channels to Images")

    tester.test(lambda: collate_channel_images(config, np.array(channels_to_images(config, channels))),
                "Collate Channel Images",
                verbose,
                lambda result, title: show_image(result))
# %%  -------------------------------------------------------------------------


"""
visualize:
  colormaps:
    food: "copper"
    poison: "summer"
    obstacle: "binary"
"""
def load_check_config(config_object):
    if isinstance(config_object, str):
        with open(config_object) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    elif isinstance(config_object, dict):
        config = config_object
    else:
        raise TypeError("config_object must be either a path (str) or a config (dict)")
    
    # asserts colormaps are of right length
    assert len(config["visualize"]["colormaps"]) == len(config["environment"]["channels"])
    # asserts colormaps are real matplotlib colormaps
    for cmap in config["visualize"]["colormaps"].values():
        assert cmap in plt.colormaps()

    return config

# Test Load Check Config
# %% --------------------------------------------------------------------------
if __name__ == "__main__":
    tester.test(lambda:
                load_check_config("./config.yaml"),
                "Load and check the configuration",
                verbose)
# %% --------------------------------------------------------------------------
