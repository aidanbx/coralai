import taichi as ti
from ...substrate.substrate import Substrate
from ...simulation.visualization import Visualization


@ti.data_oriented
class MinimalVis(Visualization):
    def __init__(
        self,
        substrate: Substrate,
        chids: list,
        name: str = None,
        scale: int = None,
    ):
        super(MinimalVis, self).__init__(substrate, chids, name, scale)

        if self.chindices.n != 1:
            raise ValueError("Vis: ch_cmaps must have 1 channel for black and white visualization")


    def render_opt_window(self):
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(200 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.channel_to_paint = 0
            self.val_to_paint = sub_w.slider_float("Value to Paint", self.val_to_paint, 0.0, 1.0)
            self.brush_radius = sub_w.slider_int("Brush Radius", self.brush_radius, 1, 200)
            self.paused = sub_w.checkbox("Pause", self.paused)
            self.perturbing_weights = sub_w.checkbox("Perturb Weights", self.perturbing_weights)
            self.perturbation_strength = sub_w.slider_float("Perturbation Strength", self.perturbation_strength, 0.0, 5.0)
       