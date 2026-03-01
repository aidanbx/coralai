"""
Non-homogeneous start environments for coral_dev.

Each class wraps spatial mask logic (where 0 = uninhabitable, 1 = habitable)
and implements the StartEnvironment interface. Replaces the old mask-based
functions in env.py.

Usage:
    from environments import make_env
    env = make_env("hole", param=0.35)
    env.seed(substrate)          # called once after evolver is ready
    if env.persist:
        env.step(substrate)      # called each step to keep barren zones dead
"""

import torch

from coralai.environment import StartEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_mask(w, h, cx, cy, radius, device):
    """Binary mask: 1 outside circle, 0 inside (hole). radius in pixels."""
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    return (dist_sq > radius ** 2).float()


def _apply_mask(substrate, mask):
    """Zero energy, infra, and genome where mask==0 (barren cells)."""
    inds = substrate.ti_indices[None]
    inv = (1.0 - mask).bool()
    substrate.mem[0, inds.energy].masked_fill_(inv, 0.0)
    substrate.mem[0, inds.infra].masked_fill_(inv, 0.0)
    substrate.mem[0, inds.genome].masked_fill_(inv, -1.0)


def _seed_habitable(substrate, mask, energy=0.4, infra=0.4):
    """Add energy and infra only where mask==1."""
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.energy].add_(mask * energy).clamp_(0.01, 100)
    substrate.mem[0, inds.infra].add_(mask * infra).clamp_(0.01, 100)


# ---------------------------------------------------------------------------
# Flat (default)
# ---------------------------------------------------------------------------

class FlatEnvironment(StartEnvironment):
    """Homogeneous environment — no spatial structure. seed and step are no-ops."""
    name = "flat"


# ---------------------------------------------------------------------------
# Hole / Ring (circular barren centre)
# ---------------------------------------------------------------------------

class HoleEnvironment(StartEnvironment):
    """Circular barren zone in the grid centre.

    param radius_frac: fraction of min(W,H)/2 that is barren (default 0.35).
    """
    name = "hole"

    def __init__(self, radius_frac: float = 0.35):
        self.radius_frac = radius_frac
        self._mask_cache = None

    def _mask(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        if self._mask_cache is None or self._mask_cache.shape != (w, h):
            radius = self.radius_frac * min(w, h) / 2.0
            self._mask_cache = _circle_mask(w, h, w / 2.0, h / 2.0,
                                            radius, device)
        return self._mask_cache

    def seed(self, substrate):
        mask = self._mask(substrate)
        _apply_mask(substrate, mask)
        _seed_habitable(substrate, mask)

    def step(self, substrate):
        _apply_mask(substrate, self._mask(substrate))

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "radius_frac": self.radius_frac}


class RingEnvironment(HoleEnvironment):
    """Same as HoleEnvironment (barren centre, ring of life around it)."""
    name = "ring"


# ---------------------------------------------------------------------------
# Stripes (alternating vertical bands)
# ---------------------------------------------------------------------------

class StripesEnvironment(StartEnvironment):
    """Alternating habitable / barren vertical bands.

    param n_stripes: total number of stripes (default 5); even-indexed = habitable.
    """
    name = "stripes"

    def __init__(self, n_stripes: int = 5):
        self.n_stripes = max(1, int(n_stripes))
        self._mask_cache = None

    def _mask(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        if self._mask_cache is None or self._mask_cache.shape != (w, h):
            n = self.n_stripes
            xs = torch.arange(w, device=device, dtype=torch.float32)
            stripe = (xs * n / w).long().clamp(0, n - 1)
            mask_1d = (stripe % 2 == 0).float()
            # Expand to (w, h) — mask is indexed [x, y]
            self._mask_cache = mask_1d.view(w, 1).expand(w, h)
        return self._mask_cache

    def seed(self, substrate):
        mask = self._mask(substrate)
        _apply_mask(substrate, mask)
        _seed_habitable(substrate, mask)

    def step(self, substrate):
        _apply_mask(substrate, self._mask(substrate))

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "n_stripes": self.n_stripes}


# ---------------------------------------------------------------------------
# Corners (four diagonal barren corners)
# ---------------------------------------------------------------------------

class CornersEnvironment(StartEnvironment):
    """Four diagonal barren corners; life fills a central diamond.

    param corner_frac: fraction of side removed per corner (default 0.25).
    """
    name = "corners"

    def __init__(self, corner_frac: float = 0.25):
        self.corner_frac = max(0.01, min(0.49, float(corner_frac)))
        self._mask_cache = None

    def _mask(self, substrate):
        mem = substrate.mem
        w, h = mem.shape[2], mem.shape[3]
        device = substrate.torch_device
        if self._mask_cache is None or self._mask_cache.shape != (w, h):
            frac = self.corner_frac
            ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h).expand(w, h)
            xs = torch.arange(w, device=device, dtype=torch.float32).view(w, 1).expand(w, h)
            tl = (xs / w + ys / h) < (frac * 2)
            tr = ((w - 1 - xs) / w + ys / h) < (frac * 2)
            bl = (xs / w + (h - 1 - ys) / h) < (frac * 2)
            br = ((w - 1 - xs) / w + (h - 1 - ys) / h) < (frac * 2)
            self._mask_cache = (~(tl | tr | bl | br)).float()
        return self._mask_cache

    def seed(self, substrate):
        mask = self._mask(substrate)
        _apply_mask(substrate, mask)
        _seed_habitable(substrate, mask)

    def step(self, substrate):
        _apply_mask(substrate, self._mask(substrate))

    @property
    def persist(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return {"name": self.name, "corner_frac": self.corner_frac}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENVIRONMENTS = {
    "flat":    FlatEnvironment,
    "hole":    HoleEnvironment,
    "ring":    RingEnvironment,
    "stripes": StripesEnvironment,
    "corners": CornersEnvironment,
}


def make_env(env_name: str, param=None) -> StartEnvironment:
    """Instantiate a StartEnvironment by name with an optional numeric param."""
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown env {env_name!r}. Choose from: {list(ENVIRONMENTS)}")
    cls = ENVIRONMENTS[env_name]
    return cls(param) if param is not None else cls()
