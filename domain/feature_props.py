"""Combinable procedural properties for landscape features.

Each axis is sampled independently so instances emerge as unique
combinations (scale × flip × tint × style × extras).
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zones import FeatureKind


@dataclass(frozen=True, slots=True)
class FeatureProps:
    """Visual / structural variant applied at blit time."""

    # universal axes
    scale_mul: float = 1.0       # 0.7 … 1.35 relative to depth scale
    flip_x: bool = False
    hue_shift: float = 0.0       # -0.15 … 0.15 (HSV)
    sat_mul: float = 1.0         # 0.7 … 1.25
    bright_mul: float = 1.0      # 0.85 … 1.15
    alpha_mul: float = 1.0       # overall opacity 0.75 … 1.0
    y_jitter: float = 0.0        # extra vertical offset px
    # style index for multi-look kinds (0 = default)
    style: int = 0
    # kind-specific extras (encoded generically)
    extra_a: float = 0.0         # e.g. animal density, chimney height
    extra_b: float = 0.0         # e.g. damage / overgrowth
    label: str = ""              # short debug / sign label

    def describe(self) -> str:
        bits = []
        if abs(self.scale_mul - 1.0) > 0.05:
            bits.append(f"s{self.scale_mul:.2f}")
        if self.flip_x:
            bits.append("flip")
        if abs(self.hue_shift) > 0.02:
            bits.append(f"h{self.hue_shift:+.2f}")
        if self.style:
            bits.append(f"st{self.style}")
        return ",".join(bits) or "default"


# ── sampling ranges per axis ─────────────────────────────────────────

_SCALE_RANGE = (0.75, 1.30)
_HUE_RANGE = (-0.12, 0.12)
_SAT_RANGE = (0.75, 1.20)
_BRIGHT_RANGE = (0.88, 1.12)
_ALPHA_RANGE = (0.85, 1.0)
_Y_JITTER = (-8.0, 12.0)

# kind → max style variants (exclusive upper bound for randint)
_STYLE_COUNT: dict[str, int] = {
    "haus": 4,
    "bauernhof": 3,
    "baum": 4,
    "busch": 3,
    "felsen": 3,
    "industrie": 3,
    "lagerhalle": 2,
    "ruine": 3,
    "wrack": 3,
    "kapelle": 2,
    "windrad": 2,
    "tiere": 3,
    "heuballen": 2,
    "acker": 2,
    "sumpf": 2,
    "laterne": 2,
    "hochspannung": 2,
}


def sample_props(rng: random.Random, kind_value: str) -> FeatureProps:
    """Independently sample all axes → combinatorial variety."""
    n_styles = _STYLE_COUNT.get(kind_value, 1)
    return FeatureProps(
        scale_mul=rng.uniform(*_SCALE_RANGE),
        flip_x=rng.random() < 0.35,
        hue_shift=rng.uniform(*_HUE_RANGE),
        sat_mul=rng.uniform(*_SAT_RANGE),
        bright_mul=rng.uniform(*_BRIGHT_RANGE),
        alpha_mul=rng.uniform(*_ALPHA_RANGE),
        y_jitter=rng.uniform(*_Y_JITTER),
        style=rng.randrange(n_styles),
        extra_a=rng.random(),
        extra_b=rng.random(),
    )


def apply_props_to_image(img, props: FeatureProps):
    """Post-process a PIL RGBA image according to props.

    Returns a new image (never mutates input).
    """
    from PIL import Image, ImageEnhance

    out = img.convert("RGBA")

    if props.flip_x:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # HSV tint
    if abs(props.hue_shift) > 0.01 or abs(props.sat_mul - 1.0) > 0.02 or abs(props.bright_mul - 1.0) > 0.02:
        out = _hsv_adjust(out, props.hue_shift, props.sat_mul, props.bright_mul)

    if abs(props.alpha_mul - 1.0) > 0.02:
        r, g, b, a = out.split()
        a = a.point(lambda p: int(p * props.alpha_mul))
        out = Image.merge("RGBA", (r, g, b, a))

    return out


def _hsv_adjust(img, hue_shift: float, sat_mul: float, bright_mul: float):
    """Per-pixel HSV adjust; preserves alpha; skips near-transparent."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(img, dtype=np.float32)
    rgb = arr[..., :3] / 255.0
    alpha = arr[..., 3]
    # vectorized approx via colorsys is slow; use simple matrix-ish approach
    # Convert RGB→HSV manually
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-9), 0.0)
    # hue
    rc = (maxc - r) / (maxc - minc + 1e-9)
    gc = (maxc - g) / (maxc - minc + 1e-9)
    bc = (maxc - b) / (maxc - minc + 1e-9)
    h = np.zeros_like(maxc)
    h = np.where((maxc == r) & (maxc != minc), (bc - gc) % 6.0, h)
    h = np.where((maxc == g) & (maxc != minc), 2.0 + rc - bc, h)
    h = np.where((maxc == b) & (maxc != minc), 4.0 + gc - rc, h)
    h = h / 6.0  # 0..1

    h = (h + hue_shift) % 1.0
    s = np.clip(s * sat_mul, 0.0, 1.0)
    v = np.clip(v * bright_mul, 0.0, 1.0)

    # HSV→RGB
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6
    rr = np.choose(i_mod, [v, q, p, p, t, v])
    gg = np.choose(i_mod, [t, v, v, q, p, p])
    bb = np.choose(i_mod, [p, p, t, v, v, q])

    # only tint opaque-ish pixels (avoid recoloring empty)
    mask = alpha > 10
    out = arr.copy()
    out[mask, 0] = rr[mask] * 255
    out[mask, 1] = gg[mask] * 255
    out[mask, 2] = bb[mask] * 255
    return Image.fromarray(out.astype(np.uint8), "RGBA")
