import time
import taichi as ti
from eincasm.Eincasm import Experiment
from eincasm.TaichiStructFactory import TaichiStructFactory


@ti.data_oriented
class Vis:
    def __init__(
        self,
        eincasm: Experiment,
        chids: list,
        name: str = None,
        scale: int = None,
    ):
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

        self.window = ti.ui.Window(
            f"{self.name}", (self.img_w, self.img_h), fps_limit=60, vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.prev_time = time.time()

        builder = TaichiStructFactory()
        builder.add_ti_i("scale", scale)
        self.var_struct = self.build_dynamic_params(builder)


    def build_dynamic_params(self, builder: TaichiStructFactory):
        """
        This is to clean up parameters that change during the simulation and thus
        have to be passed to taichi kernels
        """
        builder.add_timat_f("chlims", self.world.get_lims_timat(self.chids))
        builder.add_tivec_i("chindices", self.chindices)
        builder.add_ti_i("brush_radius", 4)
        builder.add_ti_i("chnum_to_paint", 0)
        builder.add_ti_i("chindex_to_paint", self.chindices[0])
        builder.add_ti_f("val_to_paint", 1.5)
        builder.add_ti_f("val_to_paint_dt", -1)
        builder.add_ti_i("drawing", False)
        builder.add_ti_f("mouse_posx", 0.0)
        builder.add_ti_f("mouse_posy", 0.0)
        builder.add_ti_f("perturb_strength", 0.1)
        builder.add_ti_i("is_perturbing_weights", False)
        builder.add_ti_i("is_perturbing_biases", False)
        builder.add_ti_f("test", -1)
        return builder.build()


    def update(self):
        v = self.var_struct[None]
        self.check_events(self.var_struct)
        self.render_opt_window(self.var_struct)

        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        pos = self.window.get_cursor_pos()
        v.mouse_posx = pos[0]
        v.mouse_posy = pos[1]
        v.val_to_paint_dt = v.val_to_paint * dt * 5
        if v.is_perturbing_weights:
            self.eincasm.organism.perturb_weights(v.perturb_strength)
        if v.is_perturbing_biases:
            self.eincasm.organism.perturb_biases(v.perturb_strength)

        self.update_vis(self.world.mem, self.var_struct)
        self.canvas.set_background_color((1, 1, 1))
        self.canvas.set_image(self.image)
        self.window.show()
        self.var_struct[None] = v


    @ti.kernel
    def update_vis(self, mem: ti.types.ndarray(), var_struct: ti.template()):
        v = var_struct[None]
        if v.drawing:
            self.paint_cursor_value(mem, var_struct)
        self.write_to_image(mem, var_struct)
        var_struct[None] = v


    @ti.func
    def write_to_image(self, mem: ti.types.ndarray(), var_struct: ti.template()):
        v = var_struct[None]
        for i, j in self.image:
            xind = i // v.scale
            yind = j // v.scale
            for chnum in ti.static(range(3)):
                self.image[i, j][chnum] = mem[xind, yind, v.chindices[chnum]]
        var_struct[None] = v


    @ti.func
    def paint_cursor_value(self, mem: ti.types.ndarray(), var_struct: ti.template()):
        """
        Use cursor position to paint selected value of selected channel - take into account pixel sizes
        """
        v = var_struct[None]
        ind_x = int(v.mouse_posx * mem.shape[0])
        ind_y = int(v.mouse_posy * mem.shape[1])
        for i, j in ti.ndrange(
            (-v.brush_radius, v.brush_radius), (-v.brush_radius, v.brush_radius)
        ):
            if (i**2) + j**2 < v.brush_radius**2:
                ix = (i + ind_x) % mem.shape[0]
                jy = (j + ind_y) % mem.shape[1]
                mem[ix, jy, v.chindex_to_paint] += ti.math.clamp(
                    mem[ix, jy, v.chindex_to_paint] + v.val_to_paint_dt,
                    v.chlims[v.chnum_to_paint, 0],
                    v.chlims[v.chnum_to_paint, 1],
                )
        var_struct[None] = v


    def render_opt_window(self, var_struct):
        """
        Slider for brush size
        Paint channel and value
        Any sliders for world metadata desired
        """
        v = var_struct[None]
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(180 / self.img_h, self.img_h)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as w:
            v.brush_radius = w.slider_int("Brush Radius", v.brush_radius, 1, 20)
            v.chnum_to_paint = w.slider_int("Channel to Paint", v.chnum_to_paint, 0, 2)
            v.val_to_paint = w.slider_float("Paint delta", v.val_to_paint, -2, 2)
            v.perturb_strength = w.slider_float(
                "Perturb Strength", v.perturb_strength, 0.0, 5.0
            )
            v.is_perturbing_weights = w.checkbox(
                "Perturb Weights", v.is_perturbing_weights
            )
            v.is_perturbing_biases = w.checkbox(
                "Perturb Biases", v.is_perturbing_biases
            )

            v.chindex_to_paint = v.chindices[v.chnum_to_paint]
            w.text(f"Painting: {self.chids[v.chnum_to_paint]}")
        var_struct[None] = v


    def check_events(self, var_struct):
        """
        Keys for pause and reset
        Cursor motion for painting
        """
        v = var_struct[None]
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                self.window.running = False
            # if e.key == 'p':
            #     v.is_perturbing_weights = True

        if self.window.is_pressed(ti.ui.LMB) and self.window.is_pressed(ti.ui.SPACE):
            v.drawing = True
        else:
            v.drawing = False

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                v.drawing = False
            # if e.key == 'p':
            #     v.is_perturbing_weights = False
        var_struct[None] = v
