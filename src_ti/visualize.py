import taichi as ti
import numpy as np
from matplotlib.cm import get_cmap
from src_ti.world import World
import time


LEVELS = 20
BORDER_SIZE = 2

@ti.data_oriented
class PixelVis:
    def __init__(self, world: World, ch_cmaps: dict, update_world: callable, scale=None, out_path=None):
        self.world = world
        if scale is None:
            max_dim = max(world.shape)
            desired_max_dim = 800 // 2  # PixelVis uses 2 pixels per cell
            scale = desired_max_dim // max_dim

        self.scale = scale
        self.update_world = update_world
        self.chs = self.validate_channels(ch_cmaps)
        self.chids = self.get_channel_ids()
        self.cmaps = self.process_ch_cmaps(ch_cmaps)
        self.cluster_size = 2 * self.scale + BORDER_SIZE
        self.img_w = self.world.shape[0] * self.cluster_size + 1
        self.img_h = self.world.shape[1] * self.cluster_size + 1
        self.ch_lims = self.get_channel_limits()
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))
        self.window = ti.ui.Window("PixelVis", (self.img_w, self.img_h), fps_limit=60, vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.vid_manager = ti.tools.VideoManager("./pixelvis.mp4")
        self.drawing = False
        self.ch_to_paint = 3
        self.val_to_paint = 0
        self.prev_time = time.time()
        self.brush_radius = 2

    def launch(self):
        while self.window.running:
            if not self.drawing:
                self.update_world()
            self.update()
            # self.vid_manager.write_frame(self.canvas)
        # self.vid_manager.make_video(gif=True, mp4=True)

    def update(self):
        self.check_events()
        self.render_opt_window()
        # self.vid_manager.make_video(gif=True, mp4=True)

        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        pos = self.window.get_cursor_pos()
        vtp = self.val_to_paint * dt * 5

        self.update_vis(self.world.mem,
                        self.drawing, pos[0], pos[1],
                        self.chids[self.ch_to_paint], vtp,
                        self.brush_radius//2, self.ch_lims)
        self.canvas.set_background_color((1, 1, 1))
        self.canvas.set_image(self.image)
        self.window.show() 

    @ti.kernel
    def update_vis(self, mem: ti.types.ndarray(),
                    drawing: ti.i32,
                    pos_x: ti.f32, pos_y: ti.f32,
                    chid: ti.i32, val: ti.f32, br: ti.i32, chlims: ti.types.template()):
        if drawing:
            self.paint_cursor_value(mem, pos_x, pos_y, chid, val, br, chlims)
        self.write_to_image(mem)

    @ti.func
    def write_to_image(self, mem: ti.types.ndarray()):
        for i,j in self.image:
            xind = i//self.cluster_size
            yind = j//self.cluster_size

            in_border = (i % self.cluster_size < BORDER_SIZE) or (j % self.cluster_size < BORDER_SIZE)
            if in_border:
                self.image[i, j] = ti.Vector([0.25, 0.25, 0.25])  # Black for border
            else:
                # Adjusted calculation for channel index
                adjusted_i = (i % self.cluster_size) - BORDER_SIZE
                adjusted_j = (j % self.cluster_size) - BORDER_SIZE
                chx = adjusted_i // self.scale
                chy = adjusted_j // self.scale
                chidx = chx + 2 * chy
                chid = self.chids[chidx]
                low = self.ch_lims[chidx, 0]
                high = self.ch_lims[chidx, 1]
                chval = ti.math.clamp(mem[xind, yind, chid], low, high)
                chval_norm = (chval - low) / (high - low)
                self.image[i, j] = self.cmaps[chidx, int(chval_norm * (LEVELS - 1))]

    @ti.func
    def paint_cursor_value(self,
                           mem: ti.types.ndarray(),
                           pos_x: ti.f32, pos_y: ti.f32,
                           chid: ti.i32, val: ti.f32,
                           br: ti.i32, chlims: ti.types.template()):
        """
        Use cursor position to paint selected value of selected channel - take into account pixel sizes
        """
        ind_x = int(pos_x * mem.shape[0])
        ind_y = int(pos_y * mem.shape[1])
        for i, j in ti.ndrange((-br, br), (-br, br)):
            if (i**2) + j**2 < br**2:
                ix = (i + ind_x) % mem.shape[0]
                jy = (j + ind_y) % mem.shape[1]
                mem[ix, jy, chid] = ti.math.clamp(mem[ix, jy, chid] + val, chlims[chid, 0], chlims[chid, 1])
    
    def render_opt_window(self):
        """
        Slider for brush size
        Paint channel and value
        Any sliders for world metadata desired
        """
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(180 / self.img_h, self.img_h)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as w:
            self.brush_radius = w.slider_int("Brush Radius", self.brush_radius, 1, 20)
            self.ch_to_paint = w.slider_int("Channel to Paint", self.ch_to_paint, 0, 3)
            self.val_to_paint = w.slider_float("Paint delta", self.val_to_paint, -2, 2)
            w.text(f"Painting: {self.chs[self.ch_to_paint]}")

    def check_events(self):
        """
        Keys for pause and reset
        Cursor motion for painting
        """
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                self.window.running = False
        
        if self.window.is_pressed(ti.ui.LMB) and self.window.is_pressed(ti.ui.SPACE):
            self.drawing = True
        else:
            self.drawing = False

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False


    def process_ch_cmaps(self, ch_cmaps):
        ti_cmaps = ti.Vector.field(n=3, dtype=ti.f32, shape=(4, LEVELS))
        cmap_arrs = np.zeros((4, LEVELS, 3), dtype=np.float32)
        for i in range(4):
            cmap = ch_cmaps[self.chs[i]]
            cmap_arrs[i, :, :] = get_cmap(cmap)(np.linspace(0, 1, LEVELS))[:, :3]
        ti_cmaps.from_numpy(cmap_arrs)
        return ti_cmaps
    
    def validate_channels(self, ch_cmaps):
        chs = list(ch_cmaps.keys())
        if len(chs) != 4:
            raise ValueError("PixelVis: currently only supports exactly 4 channels. Pass None to color black.")
        for ch in chs:
            if len(self.world.indices[ch]) != 1: 
                raise ValueError(f"PixelVis: currently only supports channels with exactly 1 index. Channel {ch} has {len(self.world.indices[ch])} indices. Please pass singular subchannels")
        return chs

    def get_channel_ids(self):
        chids = np.array(self.world.indices[self.chs], dtype=np.int32)
        if len(chids) != 4:
            raise ValueError("PixelVis: Oops, Error in indexing channels via world, got more than 4 Channel IDs - should have been caught above")
        chids_field = ti.field(dtype=ti.i32, shape=4)
        chids_field.from_numpy(chids)
        return chids_field

    def get_channel_limits(self):
        ch_lims = np.zeros((4,2), dtype=np.float32)
        for i, ch in enumerate(self.chs):
            ch_lims[i] = self.world.channels[ch].lims
        ch_lims_field = ti.field(dtype=ti.f32, shape=(4,2))
        ch_lims_field.from_numpy(ch_lims)
        return ch_lims_field