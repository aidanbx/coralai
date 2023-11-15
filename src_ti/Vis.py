import taichi as ti
import numpy as np
from matplotlib.cm import get_cmap
from src_ti.world import World
from src_ti.eincasm import eincasm as Eincasm
from src_ti.TaichiStructFactory import TaichiStructFactory
import time

@ti.data_oriented
class Vis:
    def __init__(self, eincasm: Eincasm, chids: list, name: str=None, scale: int = None, out_path=None):
        self.eincasm = eincasm
        self.world = eincasm.world
        self.chids = chids
        if name is None:
            name = "Vis_" + str(time.time())
        self.name = name

        chindices = self.world.get_inds_tivec(self.chids)
        if chindices.n != 3:
            raise ValueError("Vis: ch_cmaps must have 3 channels")
        self.chindices = chindices

        if scale is None:
            max_dim = max(self.world.shape)
            desired_max_dim = 800  
            scale = desired_max_dim // max_dim

        self.img_w = self.world.shape[0] * scale
        self.img_h = self.world.shape[1] * scale
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

        self.window = ti.ui.Window(f"{self.name}", (self.img_w, self.img_h), fps_limit=60, vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.prev_time = time.time()

        builder = TaichiStructFactory()
        builder.add_ti('scale', scale)
        self.vars = self.build_dynamic_params(builder)

    def build_dynamic_params(self, builder: TaichiStructFactory):
        """
        This is to clean up parameters that change during the simulation and thus
        have to be passed to taichi kernels
        """
        builder.add_ti('chlims', self.world.get_lims_tivec(self.chids))
        builder.add_ti('brush_radius', 4)
        builder.add_ti('chnum_to_paint', 0)
        builder.add_ti('chindex_to_paint', self.chindices[0])
        builder.add_ti('val_to_paint', 1.5)
        builder.add_ti('drawing', False)
        builder.add_ti('mouse_posx', 0.0)
        builder.add_ti('mouse_posy', 0.0)
        builder.add_ti('perturb_strength', 0.1)
        builder.add_ti('is_perturbing', False)
        return builder.build()

    def update(self):
        self.check_events(self.vars)
        self.render_opt_window(self.vars)

        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        pos = self.window.get_cursor_pos()
        vtp = self.val_to_paint * dt * 5
        if self.is_perturbing:
            self.eincasm.perturb_weights()

        self.update_vis(self.world.mem, self.vars)
        self.canvas.set_background_color((1, 1, 1))
        self.canvas.set_image(self.image)
        self.window.show() 

    @ti.kernel
    def update_vis(self, mem: ti.types.ndarray(), vars: ti.template()):
        if vars.drawing:
            self.paint_cursor_value(mem, vars)
        self.write_to_image(mem)


    @ti.func
    def write_to_image(self, mem: ti.types.ndarray(), vars: ti.template()):
        for i, j in self.image:
            xind = i//vars.scale 
            yind = j//vars.scale
            for chnum in ti.static(range(3)):
                self.image[i, j][chnum] = mem[xind, yind, self.chindices[chnum]]

    @ti.func
    def paint_cursor_value(self, mem: ti.types.ndarray(), vars: ti.template()):
        """
        Use cursor position to paint selected value of selected channel - take into account pixel sizes
        """
        ind_x = int(vars.mouse_posx * mem.shape[0])
        ind_y = int(vars.mouse_posy * mem.shape[1])
        for i, j in ti.ndrange((-vars.brush_radius, vars.brush_radius), (-vars.brush_radius, vars.brush_radius)):
            if (i**2) + j**2 < vars.brush_radius**2:
                ix = (i + ind_x) % mem.shape[0]
                jy = (j + ind_y) % mem.shape[1]
                mem[ix, jy, vars.chindex_to_paint] = ti.math.clamp(
                    mem[ix, jy, vars.chindex_to_paint] + vars.val_to_paint,
                    self.chlims[vars.chnum_to_paint, 0], self.chlims[vars.chnum_to_paint, 1])
    
    def render_opt_window(self, vars):
        """
        Slider for brush size
        Paint channel and value
        Any sliders for world metadata desired
        """
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(180 / self.img_h, self.img_h)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as w:
            vars.brush_radius = w.slider_int("Brush Radius", vars.brush_radius, 1, 20)
            vars.chnum_to_paint = w.slider_int("Channel to Paint", vars.chnum_to_paint, 0, 2)
            vars.val_to_paint = w.slider_float("Paint delta", vars.val_to_paint, -2, 2)
            vars.perturb_strength = w.slider_float("Perturb Strength", vars.perturb_strength, 0.0, 5.0)
            self.eincasm.perturb_strength = vars.perturb_strength
            vars.chindex_to_paint = vars.chindices[vars.chnum_to_paint]
            w.text(f"Painting: {self.chids[vars.chnum_to_paint]}")

    def check_events(self, vars):
        """
        Keys for pause and reset
        Cursor motion for painting
        """
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                self.window.running = False
            if e.key == 'p':
                vars.is_perturbing = True
        
        if self.window.is_pressed(ti.ui.LMB) and self.window.is_pressed(ti.ui.SPACE):
            vars.drawing = True
        else:
            vars.drawing = False

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                vars.drawing = False
            if e.key == 'p':
                vars.is_perturbing = False