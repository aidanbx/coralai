import taichi as ti
from src_ti.world import World


@ti.data_oriented
class PixelVis:
    def __init__(self, world: World, channels: dict, scale=1):
        self.world = world
        self.channels = channels
        self.img_w = self.world.shape[0] * 2 * scale # Pixel chunks are 2x2
        self.img_h = self.world.shape[1] * 2 * scale
        
        self.window = ti.ui.Window("PixelVis",
                              (self.img_w, self.img_h),
                              fps_limit=60, vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

    @ti.kernel
    def write_to_image(self):
        pass 

