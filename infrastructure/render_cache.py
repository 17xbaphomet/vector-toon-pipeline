"""Render-side caches to avoid re-rasterizing / re-tinting every frame."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image


class LRU:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(8, capacity)
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# Shared process-wide caches
_svg_cache: LRU = LRU(128)          # path → RGBA Image (base raster)
_variant_cache: LRU = LRU(256)      # (path, props_key) → RGBA after props
_size_cache: LRU = LRU(384)         # (path, props_key, w, h) → resized RGBA
_moon_cache: LRU = LRU(24)          # phase_bucket → moon sprite
_sky_cache: LRU = LRU(16)           # (w, h, top, bot) → sky image


def trim_alpha(img: Image.Image, threshold: int = 8) -> Image.Image:
    """Crop to non-transparent content so the visual bottom sits on the ground."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    mask = img.getchannel("A").point(lambda a: 255 if a > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


def props_key(props) -> tuple:
    if props is None:
        return ()
    # quantize floats so near-identical props share cache entries
    return (
        round(props.scale_mul, 2),
        props.flip_x,
        round(props.hue_shift, 2),
        round(props.sat_mul, 2),
        round(props.bright_mul, 2),
        round(props.alpha_mul, 2),
        props.style,
    )


def get_svg_rgba(path: Path, svg_to_pil) -> Image.Image:
    key = str(path)
    hit = _svg_cache.get(key)
    if hit is not None:
        return hit
    img = svg_to_pil(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    _svg_cache.put(key, img)
    return img


def get_variant_rgba(path: Path, props, svg_to_pil, apply_props) -> Image.Image:
    pk = props_key(props)
    key = (str(path), pk)
    hit = _variant_cache.get(key)
    if hit is not None:
        return hit
    base = get_svg_rgba(path, svg_to_pil)
    if props is None or pk == ():
        out = base
    else:
        out = apply_props(base, props)
    _variant_cache.put(key, out)
    return out


def get_sized_rgba(
    path: Path, props, w: int, h: int, svg_to_pil, apply_props
) -> Image.Image:
    """Scale feature into (w, h) box, preserve aspect, trim transparent padding.

    Prefer filling the requested width; if resulting height would exceed `h`,
    scale down so the content fits inside the box. Never stretches.
    Returned size may be smaller than (w, h) — caller must bottom-align to ground.
    """
    w = max(1, int(w))
    h = max(1, int(h))
    wq = max(1, (w // 4) * 4)
    hq = max(1, (h // 4) * 4)
    pk = props_key(props)
    key = (str(path), pk, wq, hq, "trim_v2")
    hit = _size_cache.get(key)
    if hit is not None:
        if hit.size[0] == w and hit.size[1] <= h:
            return hit
        scale = w / max(1, hit.size[0])
        nw = w
        nh = max(1, int(round(hit.size[1] * scale)))
        if nh > h:
            scale = h / max(1, hit.size[1])
            nw = max(1, int(round(hit.size[0] * scale)))
            nh = h
        return hit.resize((nw, nh), Image.Resampling.BILINEAR)

    art = get_variant_rgba(path, props, svg_to_pil, apply_props)
    art = trim_alpha(art)
    ow, oh = art.size
    if ow < 1 or oh < 1:
        empty = Image.new("RGBA", (w, 1), (0, 0, 0, 0))
        _size_cache.put(key, empty)
        return empty

    # Prefer fill width; if too tall, fit to height instead (never stretch).
    scale = wq / ow
    nw = wq
    nh = max(1, int(round(oh * scale)))
    if nh > hq:
        scale = hq / oh
        nw = max(1, int(round(ow * scale)))
        nh = hq
    sized = art.resize((nw, nh), Image.Resampling.BILINEAR)
    _size_cache.put(key, sized)
    if nw != w or nh != h:
        scale_w = w / max(1, nw)
        scale_h = h / max(1, nh)
        scale = min(scale_w, scale_h)
        fw = max(1, int(round(nw * scale)))
        fh = max(1, int(round(nh * scale)))
        if (fw, fh) != (nw, nh):
            return sized.resize((fw, fh), Image.Resampling.BILINEAR)
    return sized


def get_sky(w: int, h: int, top: tuple, bot: tuple) -> Image.Image:
    key = (w, h, top, bot)
    hit = _sky_cache.get(key)
    if hit is not None:
        return hit.copy()
    import numpy as np

    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    mid = 0.55
    u1 = np.clip(t / mid, 0, 1)
    r = top[0] + (bot[0] - top[0]) * u1
    g = top[1] + (bot[1] - top[1]) * u1
    b = top[2] + (bot[2] - top[2]) * u1
    mask2 = t >= mid
    u2 = (t - mid) / (1.0 - mid)
    r = np.where(mask2, bot[0] * (1 - 0.15 * u2), r)
    g = np.where(mask2, bot[1] * (1 - 0.15 * u2), g)
    b = np.where(mask2, bot[2] * (1 - 0.15 * u2), b)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    arr = np.repeat(arr, w, axis=1)
    img = Image.fromarray(arr, "RGB").convert("RGBA")
    _sky_cache.put(key, img)
    return img.copy()


def get_moon_sprite(phase: float, radius: int = 22) -> Image.Image:
    bucket = int(round(phase * 32)) % 33
    key = (bucket, radius)
    hit = _moon_cache.get(key)
    if hit is not None:
        return hit
    import math

    r = radius
    moon_lit = (230, 230, 210, 240)
    moon_dark = (230, 230, 210, 18)
    pa = (bucket / 32.0) * 2.0 * math.pi
    moon_img = Image.new("RGBA", (r * 2 + 2, r * 2 + 2), (0, 0, 0, 0))
    cx, cy = r + 1, r + 1
    pixels = moon_img.load()
    for py in range(r * 2 + 2):
        dy = py - cy
        dy2 = dy * dy
        if dy2 > r * r:
            continue
        half = math.sqrt(r * r - dy2)
        term = -math.cos(pa) * half
        for px in range(r * 2 + 2):
            dx = px - cx
            if dx * dx + dy2 > r * r:
                continue
            pixels[px, py] = moon_lit if dx >= term else moon_dark
    _moon_cache.put(key, moon_img)
    return moon_img


def cache_stats() -> dict[str, int]:
    return {
        "svg": len(_svg_cache),
        "variant": len(_variant_cache),
        "size": len(_size_cache),
        "moon": len(_moon_cache),
        "sky": len(_sky_cache),
    }
