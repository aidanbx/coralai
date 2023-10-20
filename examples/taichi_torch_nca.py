import torch
import torch.nn as nn
import taichi as ti

# arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.metal)
img_w, img_h = 512, 256
cell_size = 1
w, h = img_w // cell_size, img_h // cell_size
render_buffer = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=(img_w, img_h)
        )

@ti.kernel
def write_to_renderer(
        state: ti.types.ndarray(dtype=ti.f32),
    ):
    for i, j in ti.ndrange(img_w, img_h):
        for p in ti.static(range(3)):
            render_buffer[i, j][p] = state[0, p, h - j//cell_size, i//cell_size]

@ti.kernel
def add_noise(strength: ti.f32, state: ti.types.ndarray()):
    for i, j in ti.ndrange(w, h):
        for ch in ti.static(range(3)):
            state[0, ch, i, j] += ti.random(float) * strength

@ti.kernel
def set_neigh_zero(
        ind_x: ti.i32,
        ind_y: ti.i32,
        radius: ti.i32,
        state: ti.types.ndarray()
    ):
    for i, j in ti.ndrange((ind_x-radius, ind_x+radius), (ind_y-radius, ind_y+radius)):
        for ch in ti.static(range(3)):
            if (i-ind_x)**2 + 2*(j-ind_y)**2 < radius**2:
                state[0, ch, i, j] *= 0.001

class NCA(nn.Module):
    def __init__(self, channel_count):
        super(NCA, self).__init__()
        self.state = torch.rand(1, channel_count, w, h)
        self.conv = nn.Conv2d(
            channel_count,
            channel_count,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )

        self.paused = False
        self.brush_radius = 30
        self.drawing = False
        self.perturbing_weights = False
        self.perturbation_strength = 0.1
        self.noise_strength = 0.01
        print(self.state.shape)

    def perturb_weights(self):
        self.conv.weight.data[3:, 3:, :, :] += torch.randn_like(self.conv.weight.data[3:, 3:, :, :]) * self.perturbation_strength

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

    def apply_rules(self):
        self.state = self.forward(self.state)

    # Define the update function for the animation
    def update(self, window):
        add_noise(self.noise_strength, self.state)
        if self.drawing:
            pos = window.get_cursor_pos()
            index_y = int(max(self.brush_radius, min(pos[0] * w, w - self.brush_radius)))
            index_x = h - int(max(self.brush_radius, min(pos[1] * h, h - self.brush_radius)))
            set_neigh_zero(index_x, index_y, self.brush_radius, self.state)

        if self.perturbing_weights:
            self.perturb_weights()
        write_to_renderer(self.state)
    
    def check_input(self, window):
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and window.is_pressed(ti.ui.SHIFT):
                self.drawing = True
            # elif e.key == ti.ui.SPACE:
            #     self.paused = not self.paused
            # elif e.key == 'r':
            #     self.perturbing_weights = True

        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False
            # elif e.key == 'r':
            #     self.perturbing_weights = False

        
def main(img_w,img_h):
    model = NCA(5)
    window = ti.ui.Window("NCA", (img_w, img_h), fps_limit=200, vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    steps_per_frame = 1

    while window.running:
        model.check_input(window)

        if not model.paused:
            for _ in range(steps_per_frame):
                model.apply_rules()
            model.update(window)

        canvas.set_background_color((1, 1, 1))
        with gui.sub_window("Options", 0.05, 0.05, 0.9, 0.15) as w:
            model.brush_radius = w.slider_int("Brush Radius", model.brush_radius, 1, 50)
            model.noise_strength = w.slider_float("Noise Strength", model.noise_strength, 0.0, 5.0)
            model.perturbation_strength = w.slider_float("Perturbation Strength", model.perturbation_strength, 0.0, 5.0)
            steps_per_frame = w.slider_int("Steps per Frame", steps_per_frame, 1, 100)
            model.paused = w.checkbox("Pause", model.paused)
            model.perturbing_weights = w.checkbox("Perturb Weights", model.perturbing_weights)
        canvas.set_image(render_buffer)
        window.show()

if __name__ == "__main__":
    main(img_w, img_h)
