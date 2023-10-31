import taichi as ti
import numpy as np
from matplotlib.cm import get_cmap
from src_ti.world import World

DEFAULT_CMAPS = [
    ('viridis', 10),
    ('copper', 10),
    ('hot', 10),
    ('Greys', 20),
]

@ti.data_oriented
class PixelVis:
    def __init__(self, world: World, chs: list, cmaps_chunks: list=None, scale=1):
        self.world = world
        if len(chs) != 4:
            raise ValueError("PixelVis: currently only supports exactly 4 channels. Pass None to color black.")
        for ch in chs:
            if len(world.index[ch]) != 1: 
                raise ValueError(f"PixelVis: currently only supports channels with exactly 1 index. Channel {ch} has {len(world.index[ch])} indices. Please pass singular subchannels")
        self.chids = world.index[chs]
        if len(self.chids) != 4:
            raise ValueError("PixelVis: Oops, Error in indexing channels via world, got more than 4 Channel IDs - should have been caught above")
        

            
        self.cmaps = self.process_ch_cmaps(self.chs, cmaps_chunks)
        self.img_w = self.world.shape[0] * 2 * scale # Pixel chunks are scale*(2,2)
        self.img_h = self.world.shape[1] * 2 * scale
        self.scale = scale
        
        self.window = ti.ui.Window("PixelVis",
                              (self.img_w, self.img_h),
                              fps_limit=60, vsync=True)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

    def update(self):
        self.check_events()
        self.paint_cursor_value()
        self.render_opt_window()
        self.write_to_image()
        self.canvas.set_image(self.image)
        self.window.show() 

    @ti.func
    def get_field_name_from_index(self, index: ti.i32) -> ti.i32:
        return self.chs_indices[index]

    @ti.kernel
    def write_to_image(self):
        for i,j in self.image:
            xind = i//(2*self.scale)
            yind = j//(2*self.scale)
            chx = (i%(2*self.scale))//self.scale # (0-1)
            chy = (j%(2*self.scale))//self.scale # (0-1)
            chidx = self.get_field_name_from_index(chx + 2*chy)
            chid = self.chs[chidx]
            cmap = self.cmaps[chidx]
            if ti.static(chid is None):
                chval = 0
            else:
                chval = self.world.data[xind, yind][chid]
            chval_norm = (chval - self.world[chid]['lims'][0]) / (self.world[chid]['lims'][1] - self.world[chid]['lims'][0])
            self.image[i, j] = cmap[int(chval_norm * (len(cmap) - 1))]

    def render_opt_window(self):
        """
        Slider for brush size
        Paint channel and value
        Any sliders for world metadata desired
        """
        pass

    def check_events(self):
        """
        Keys for pause and reset
        Cursor motion for painting
        """
        pass

    def paint_cursor_value(self):
        """
        Use cursor position to paint selected value of selected channel - take into account pixel sizes
        """
        pass

    @staticmethod
    def _gen_cmap_array(cmap_name, n_colors):
        cmap = get_cmap(cmap_name)
        return cmap(np.linspace(0, 1, n_colors))[:, :3]
    
    def process_ch_cmaps(self, chs, cmaps_chunks=None):
        # TODO: INCOMPLETE, NEED TO TURN INTO TAICHI FIELD FOR RENDERING INDEXING MAGIC
        if cmap_chunks is None:
            cmap_chunks = [None] * len(chs)
        if len(cmaps_chunks) != 4:
            raise ValueError(f"PixelVis: Expected 4 cmap chunks, got {len(cmaps_chunks)}")
        new_cmaps = []
        for i in range(len(cmaps_chunks)):
            cmap = cmaps_chunks[i]
            chid = chs[i]
            n_colors = 10

            if chid is None:
                cmap = 'grey'
                n_colors = 1
            elif cmap is None:
                cmap = DEFAULT_CMAPS[i][0]
                n_colors = DEFAULT_CMAPS[i][1]
            elif isinstance(cmap, tuple):
                n_colors = cmap[1]
                cmap = cmap[0]
            elif not isinstance(cmap, str):
                raise ValueError(f"PixelVis: For Channel: ({chid}), Given cmap: ({cmap}) which is not a string or tuple. Pass as (\"cmap\", n_colors) or \"cmap\"")
            
            new_cmaps.append(self._gen_cmap_array(cmap, n_colors))
        
        ti_cmaps = ti.Vector.field(n=3, dtype=ti.f32, shape=(4, 10))
        return new_cmaps
    