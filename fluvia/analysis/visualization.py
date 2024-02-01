from ..substrate.world import World


class Visualization:
    def __init__(self,
                 world: World,
                 chids: list,
                 name: str = None,
                 scale: int = None,):
        self.world = world
        self.w = world.w
        self.h = world.h
        self.chids = chids
        self.name = "Vis" if name is None else name
        self.scale = 1 if scale is None else scale


    def update(self):
        pass