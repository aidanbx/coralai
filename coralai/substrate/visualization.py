import time
import torch
import taichi as ti
from .substrate import Substrate


@ti.data_oriented
class Visualization:
    def __init__(self,
                 world: Substrate,
                 chids: list,
                 name: str = None,
                 scale: int = None,):
        self.world = world
        self.w = world.w
        self.h = world.h
        self.chids = chids
        self.name = "Vis" if name is None else name
        self.scale = 1 if scale is None else scale

        chindices = self.world.get_inds_tivec(self.chids)

        self.chindices = chindices

        if scale is None:
            max_dim = max(self.world.w, self.world.h)
            desired_max_dim = 800
            scale = desired_max_dim // max_dim
            
        self.scale = scale
        self.img_w = self.world.w * scale
        self.img_h = self.world.h * scale
        self.n_channels = len(chindices)
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

        self.window = ti.ui.Window(
            f"{self.name}", (self.img_w, self.img_h), fps_limit=200, vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.paused = False
        self.brush_radius = 4
        self.mutating = False
        self.perturbation_strength = 0.1
        self.drawing = False
        self.prev_time = time.time()
        self.prev_pos = self.window.get_cursor_pos()
        self.channel_to_paint = 0
        self.val_to_paint = 0.1


    @ti.kernel
    def add_val_to_loc(self,
            val: ti.f32,
            pos_x: ti.f32,
            pos_y: ti.f32,
            radius: ti.i32,
            channel_to_paint: ti.i32,
            mem: ti.types.ndarray()
        ):
        ind_x = int(pos_x * self.w)
        ind_y = int(pos_y * self.h)
        offset = int(pos_x) * 3
        for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
            if (i**2) + j**2 < radius**2:
                mem[0, offset+channel_to_paint, (i + ind_x) % self.w, (j + ind_y) % self.h] += val


    @ti.kernel
    def write_to_renderer(self, mem: ti.types.ndarray(), max_vals: ti.types.ndarray(), chindices: ti.template()):
        for i, j in self.image:
            xind = (i//self.scale) % self.w
            yind = (j//self.scale) % self.h
            for k in ti.static(range(3)):
                chid = chindices[k%self.n_channels]
                self.image[i, j][k] = mem[0, chid, xind, yind]


    def render_opt_window(self):
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(240 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.channel_to_paint = sub_w.slider_int("Channel to Paint", self.channel_to_paint, 0, 2)
            self.val_to_paint = sub_w.slider_float("Value to Paint", self.val_to_paint, 0.0, 1.0)
            self.brush_radius = sub_w.slider_int("Brush Radius", self.brush_radius, 1, 200)
            self.paused = sub_w.checkbox("Pause", self.paused)
            self.mutating = sub_w.checkbox("Perturb Weights", self.mutating)
            self.perturbation_strength = sub_w.slider_float("Perturbation Strength", self.perturbation_strength, 0.0, 5.0)


    def check_events(self):
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in  [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and self.window.is_pressed(ti.ui.SHIFT):
                self.drawing = True
            elif e.key == ti.ui.SPACE:
                self.world.mem *= 0.0
        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False


    def update(self):
        current_time = time.time()
        current_pos = self.window.get_cursor_pos()
        if not self.paused:
            self.check_events()
            if self.drawing and ((current_time - self.prev_time) > 0.1): # or (current_pos != self.prev_pos)):
                self.add_val_to_loc(self.val_to_paint, current_pos[0], current_pos[1], self.brush_radius, self.channel_to_paint, self.world.mem)
                self.prev_time = current_time  # Update the time of the last action
                self.prev_pos = current_pos

            max_vals = torch.tensor([self.world.mem[0, ch].max() for ch in self.chindices])
            self.write_to_renderer(self.world.mem, max_vals, self.chindices)
        self.render_opt_window()
        self.canvas.set_image(self.image)
        self.window.show()
