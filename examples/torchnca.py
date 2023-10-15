from matplotlib import gridspec, pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import torch


def torch_NCA():
    import torch
    import torch.nn as nn

    class NCAModel(nn.Module):
        def __init__(self, channel_count):
            super(NCAModel, self).__init__()
            state = torch.rand(2, n_channels, 100, 100)
            state[:,3,...]=torch.zeros_like(state[:,3,...])
            state[:,3,45:55,45:55]=torch.ones_like(state[:,3,45:55,45:55])
            self.state = state
            self.conv = nn.Conv2d(channel_count, channel_count, kernel_size=3, padding=1, padding_mode='circular')

        def forward(self, x):
            # Apply the convolutional layer
            x = self.conv(x)
            
            # Apply the activation function
            x = nn.ReLU()(x)
            # batch norm
            x = nn.BatchNorm2d(x.shape[1])(x)
            # Ensure output is within 0..1 for image data
            x = torch.sigmoid(x)

            return x

    def extract_ims(state):
        return state.detach().permute(0,2,3,1)[:,:,:,:4].numpy()
    
    # Define the update function for the animation
    def update(f, model):
        state = model(model.state)
        model.state = state
        img1, img2 = extract_ims(state)
        im1.set_array(img1)
        im2.set_array(img2)

        # add some random noise to state
        state += torch.rand_like(state)*0.1-0.05
    
        return im1, im2

    n_hidden = 7
    n_channels = 4 + n_hidden
    # Initialize the state
    
    # Create the model
    model = NCAModel(channel_count=n_channels)

    # Create the figure for the animation
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    ims = extract_ims(model.state)

    im1 = axs[0].imshow(ims[0])
    im2 = axs[1].imshow(ims[1])

    # Create the animation
    ani = FuncAnimation(fig, lambda frame: update(frame, model), frames=100, interval=50, repeat=True)

    # Display the animation
    plt.show()

if __name__ == "__main__":
    torch_NCA()