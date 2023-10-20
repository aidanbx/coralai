import taichi as ti
import torch
import numpy as np

ti.init()

device = 'mps:0'

# Shared parameters and signal generation
num_signals = 4
min_freq, max_freq = 0.1, np.pi*2
min_rot, max_rot = 0, np.pi
min_amp, max_amp = 1, 1
noise_scale = 0.05  # Added noise scale

freqs = torch.rand(num_signals) * (max_freq - min_freq) + min_freq
rots = torch.rand(num_signals) * (max_rot - min_rot) + min_rot
amps = torch.rand(num_signals) * (max_amp - min_amp) + min_amp
periods = torch.zeros(num_signals)

width, height = 200, 200
min_angle, max_angle = -np.pi, np.pi
step = (max_angle - min_angle) / 100
x = torch.linspace(min_angle, max_angle, width)
y = torch.linspace(min_angle, max_angle, height)
X, Y = torch.meshgrid(x, y)
X = X.to(device)
Y = Y.to(device)
image = torch.empty((width, height)).to(device)

def randomize_params():
    freqs = torch.rand(num_signals) * (max_freq - min_freq) + min_freq
    rots = torch.rand(num_signals) * (max_rot - min_rot) + min_rot
    amps = torch.rand(num_signals) * (max_amp - min_amp) + min_amp
    periods = torch.zeros(num_signals)

def generate_signal(freq, period, amplitude, rot, X, Y):
    noise = torch.randn_like(X) * noise_scale  # Added noise
    return (amplitude * torch.sin(freq * X * torch.cos(rot) - Y * torch.sin(rot) + period) +
            amplitude * torch.cos(freq * X * torch.sin(rot) + Y * torch.cos(rot) + period)) + noise

def udpate_weather(period):
    # 2D signals
    image = sum([generate_signal(freqs[i], period, amps[i], rots[i], X, Y) for i in range(num_signals)])

@ti.kernel
def update_image(gui: ti.types.gui(), image: ti.types.ndarray()):
    for i, j in image:
        image[i, j] = ti.sin(image[i,j])   

def main():
    gui = ti.GUI("Game of Life", (width, height))
    gui.fps_limit = 150

    print("[Hint] Press `r` to reset")
    print("[Hint] Press SPACE to pause")
    print("[Hint] Click LMB, RMB and drag to add alive / dead cells")

    paused = False
    period = min_angle*2
    while gui.running:
        for e in gui.get_events(gui.PRESS, gui.MOTION):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                paused = not paused
            elif e.key == "r":
                randomize_params()

        if not paused:
            udpate_weather(period)
            update_image(image)
            period += step

        gui.set_image(ti.from_torch(image))
        gui.show()


if __name__ == "__main__":
    main()