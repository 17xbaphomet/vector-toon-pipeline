"""MVP renderer: composite SVG layers with Pillow + cairosvg, swap mouth by viseme."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cairosvg
from PIL import Image

from domain.entities import CharacterRig, FrameState
from domain.interfaces import FrameRenderer
from domain.value_objects import Affine, Viseme


class PillowCutoutRenderer(FrameRenderer):
    """
    Renders a single frame by:
    1. Loading base / layer SVGs → PNG via cairosvg
    2. Selecting the correct mouth shape for the current viseme
    3. Applying simple root + bone transforms (translate/scale for MVP)
    4. Compositing onto a transparent or solid canvas
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
    ) -> Path:
        w, h = canvas_size
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))

        # 1. Body / base
        body = self._svg_to_pil(rig.base_svg)
        # 2. Mouth for current viseme
        mouth_path = rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        mouth = self._svg_to_pil(mouth_path) if mouth_path else None

        # Simple placement: root position is center of character
        cx, cy = state.root_position
        scale = state.scale

        def paste_centered(img: Image.Image, ox: float = 0.0, oy: float = 0.0) -> None:
            iw, ih = img.size
            # scale
            if scale != 1.0:
                nw, nh = int(iw * scale), int(ih * scale)
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                iw, ih = nw, nh
            x = int(cx + ox - iw / 2)
            y = int(cy + oy - ih / 2)
            canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)

        paste_centered(body)

        # Head bob from bone_transforms
        head_tf = state.bone_transforms.get("head", Affine.identity())
        head_oy = head_tf.f  # ty component

        if mouth is not None:
            # Mouth is roughly at head height; offset a bit upward
            paste_centered(mouth, oy=head_oy - 30 * scale)

        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        canvas.convert("RGB").save(out, "PNG")
        return out

    def _svg_to_pil(self, svg_path: Path | None) -> Image.Image:
        if svg_path is None:
            return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        key = str(svg_path.resolve())
        if key in self._png_cache:
            return self._png_cache[key].copy()

        png_bytes = cairosvg.svg2png(url=str(svg_path))
        from io import BytesIO

        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        self._png_cache[key] = img
        return img.copy()
