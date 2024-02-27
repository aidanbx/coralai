import taichi as ti
from ...substrate.substrate import Substrate
from ...simulation.visualization import Visualization


@ti.data_oriented
class CoralVis(Visualization):
    def __init__(
        self,
        substrate: Substrate,
        chids: list,
        name: str = None,
        scale: int = None,
    ):
        super(CoralVis, self).__init__(substrate, chids, name, scale)

        if self.chindices.n != 3:
            raise ValueError("Vis: ch_cmaps must have 3 channels")
        

    def render_opt_window(self):
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(200 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.channel_to_paint = sub_w.slider_int("Channel to Paint", self.channel_to_paint, 0, 2)
            self.val_to_paint = sub_w.slider_float("Value to Paint", self.val_to_paint, 0.0, 1.0)
            self.brush_radius = sub_w.slider_int("Brush Radius", self.brush_radius, 1, 200)
            self.paused = sub_w.checkbox("Pause", self.paused)
            self.mutating = sub_w.checkbox("Perturb Weights", self.mutating)
            self.perturbation_strength = sub_w.slider_float("Perturbation Strength", self.perturbation_strength, 0.0, 5.0)
            sub_w.text("Channel Stats:")
            for channel_name in ['energy', 'infra']:
                chindex = self.world.windex[channel_name]
                max_val = self.world.mem[0, chindex].max()
                min_val = self.world.mem[0, chindex].min()
                avg_val = self.world.mem[0, chindex].mean()
                sum_val = self.world.mem[0, chindex].sum()
                sub_w.text(f"{channel_name}: Max: {max_val:.2f}, Min: {min_val:.2f}, Avg: {avg_val:.2f}, Sum: {sum_val:.2f}")
                