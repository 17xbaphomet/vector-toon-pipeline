"""Frame renderer: parallax backgrounds + SVG character + mouth swap."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence

import cairosvg
from PIL import Image

from domain.entities import CharacterRig, FrameState
from domain.interfaces import FrameRenderer
from domain.procedural import parallax_offset
from domain.value_objects import Affine, BackgroundLayer, Viseme


class PillowCutoutRenderer(FrameRenderer):
    """
    1. Draw background layers (sorted by z_index) with parallax
    2. Composite character body + mouth for current viseme
    3. Apply root position + simple bone offsets
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or "./work/svg_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._png_cache: dict[str, Image.Image] = {}

    def render(
        self,
        state: FrameState,
        rig: CharacterRig,
        canvas_size: tuple[int, int],
        backgrounds: Sequence[BackgroundLayer] | None = None,
    ) -> Path:
        w, h = canvas_size
        canvas = Image.new("RGBA", (w, h), (135, 206, 235, 255))  # sky blue default

        # --- Backgrounds (back → front) ---
        if backgrounds:
            layers = sorted(backgrounds, key=lambda b: b.z_index)
            for layer in layers:
                img = self._svg_to_pil(Path(layer.path))
                ox, oy = parallax_offset(
                    state.camera,
                    layer.parallax,
                    state.time,
                    layer.scroll_x,
                    layer.scroll_y,
                )
                self._paste(canvas, img, int(ox), int(oy), tile_x=layer.repeat_x)

        # --- Character ---
        body = self._svg_to_pil(rig.base_svg)
        mouth_path = rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        mouth = self._svg_to_pil(mouth_path) if mouth_path else None

        cx, cy = state.root_position
        scale = state.scale

        def paste_char(img: Image.Image, ox: float = 0.0, oy: float = 0.0) -> None:
            local = img
            if scale != 1.0:
                nw = max(1, int(local.width * scale))
                nh = max(1, int(local.height * scale))
                local = local.resize((nw, nh), Image.Resampling.LANCZOS)
            x = int(cx + ox - local.width / 2)
            y = int(cy + oy - local.height / 2)
            canvas.paste(local, (x, y), local if local.mode == "RGBA" else None)

        paste_char(body)

        head_tf = state.bone_transforms.get("head", Affine.identity())
        head_oy = head_tf.f
        if mouth is not None:
            paste_char(mouth, oy=head_oy - 30 * scale)

        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        canvas.convert("RGB").save(out, "PNG")
        return out

    def _paste(
        self,
        canvas: Image.Image,
        img: Image.Image,
        ox: int,
        oy: int,
        tile_x: bool = False,
    ) -> None:
        """Paste image; optionally tile horizontally to fill canvas."""
        cw, ch = canvas.size
        iw, ih = img.size
        if not tile_x:
            canvas.paste(img, (ox, oy), img if img.mode == "RGBA" else None)
            return
        # tile covering [-iw, cw+iw]
        start = ox % iw - iw
        x = start
        while x < cw + iw:
            canvas.paste(img, (x, oy), img if img.mode == "RGBA" else None)
            x += iw

    def _svg_to_pil(self, svg_path: Path | None) -> Image.Image:
        if svg_path is None or not Path(svg_path).is_file():
            return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        key = str(Path(svg_path).resolve())
        if key in self._png_cache:
            return self._png_cache[key].copy()

        png_bytes = cairosvg.svg2png(url=str(svg_path))
        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        self._png_cache[key] = img
        return img.copy()
