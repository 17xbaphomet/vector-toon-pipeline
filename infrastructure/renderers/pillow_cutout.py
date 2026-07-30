"""Frame renderer: SVG torso + procedural capsule limbs (rigid joints)."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Sequence

import cairosvg
from PIL import Image, ImageDraw

from domain.entities import CharacterRig, FrameState
from domain.interfaces import FrameRenderer
from domain.procedural import HIP_HEIGHT, SHIN_LEN, THIGH_LEN, parallax_offset
from domain.value_objects import Affine, BackgroundLayer, Viseme

UPPER_ARM_LEN = 40.0
FOREARM_LEN = 38.0


def _ang(tf: Affine) -> float:
    return math.degrees(math.atan2(tf.b, tf.a))


def _dir(angle_deg: float, length: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return math.sin(rad) * length, math.cos(rad) * length


class PillowCutoutRenderer(FrameRenderer):
    """Torso SVG + capsule limbs; supports disk PNG and in-memory Image."""

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
        img = self.render_image(state, rig, canvas_size, backgrounds)
        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        img.convert("RGB").save(out, "PNG")
        return out

    def render_image(
        self,
        state: FrameState,
        rig: CharacterRig,
        canvas_size: tuple[int, int],
        backgrounds: Sequence[BackgroundLayer] | None = None,
    ) -> Image.Image:
        """Render one frame to an in-memory RGB PIL Image (for streaming)."""
        w, h = canvas_size
        canvas = Image.new("RGBA", (w, h), (135, 206, 235, 255))

        if backgrounds:
            for layer in sorted(backgrounds, key=lambda b: b.z_index):
                img = self._svg_to_pil(Path(layer.path))
                ox, oy = parallax_offset(
                    state.camera, layer.parallax, state.time,
                    layer.scroll_x, layer.scroll_y,
                )
                self._paste_tile(canvas, img, int(ox), int(oy), layer.repeat_x)

        cx, cy = state.root_position
        flip = state.scale < 0
        s = abs(state.scale) if state.scale else 1.0
        bones = state.bone_transforms
        bob = bones.get("body", Affine.identity()).f * s

        hip_x, hip_y = cx, cy - HIP_HEIGHT * s + bob
        sh_x, sh_y = cx, hip_y - 70 * s
        torso_x, torso_y = cx, hip_y - 30 * s

        side_svg = Path(rig.base_svg).parent / "body_side.svg"
        torso_path = side_svg if side_svg.is_file() else rig.base_svg
        torso = self._svg_to_pil(torso_path)
        if flip:
            torso = torso.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        leg_col = (44, 95, 138, 255)
        leg_outline = (26, 58, 85, 255)
        arm_col = (245, 203, 167, 255)
        arm_outline = (212, 165, 116, 255)
        foot_col = (26, 26, 26, 255)

        if not flip:
            order = ["right", "left"]
            dx = {"right": -5.0, "left": 5.0}
        else:
            order = ["left", "right"]
            dx = {"left": -5.0, "right": 5.0}

        far_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        near_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        for i, side in enumerate(order):
            layer = far_layer if i == 0 else near_layer
            hx = hip_x + dx[side] * s
            sx = sh_x + dx[side] * s

            th = _ang(bones.get(f"{side}_thigh") or bones.get(f"{side}_leg", Affine.identity()))
            shn = _ang(bones.get(f"{side}_shin", Affine.identity()))
            if flip:
                th, shn = -th, -shn

            knee = (hx + _dir(th, THIGH_LEN * s)[0], hip_y + _dir(th, THIGH_LEN * s)[1])
            ankle = (knee[0] + _dir(shn, SHIN_LEN * s)[0], knee[1] + _dir(shn, SHIN_LEN * s)[1])

            self._capsule(layer, (hx, hip_y), knee, 9 * s, leg_col, leg_outline)
            self._capsule(layer, knee, ankle, 7 * s, leg_col, leg_outline)
            self._foot(layer, ankle, shn, 12 * s, foot_col)

            ua = _ang(bones.get(f"{side}_upper_arm") or bones.get(f"{side}_arm", Affine.identity()))
            fa = _ang(bones.get(f"{side}_forearm", Affine.identity()))
            if flip:
                ua, fa = -ua, -fa

            elbow = (sx + _dir(ua, UPPER_ARM_LEN * s)[0], sh_y + _dir(ua, UPPER_ARM_LEN * s)[1])
            hand = (elbow[0] + _dir(fa, FOREARM_LEN * s)[0], elbow[1] + _dir(fa, FOREARM_LEN * s)[1])

            self._capsule(layer, (sx, sh_y), elbow, 6 * s, arm_col, arm_outline)
            self._capsule(layer, elbow, hand, 5 * s, arm_col, arm_outline)
            self._disk(layer, hand, 6 * s, arm_col)

        canvas = Image.alpha_composite(canvas, far_layer)
        self._blit_centered(canvas, torso, torso_x, torso_y, s)
        canvas = Image.alpha_composite(canvas, near_layer)

        mouth_path = rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        if mouth_path:
            mouth = self._svg_to_pil(mouth_path)
            if flip:
                mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            head_bob = bones.get("head", Affine.identity()).f * s
            mx = cx + (18 if not flip else -18) * s
            my = sh_y - 28 * s + head_bob
            self._blit_centered(canvas, mouth, mx, my, s * 0.65)

        return canvas.convert("RGB")

    def _capsule(self, img, p0, p1, radius, fill, outline=None) -> None:
        draw = ImageDraw.Draw(img)
        r = max(1.0, radius)
        draw.line([p0, p1], fill=fill, width=int(r * 2))
        for cx_, cy_ in (p0, p1):
            draw.ellipse([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=fill, outline=outline)

    def _foot(self, img, ankle, shin_ang, size, fill) -> None:
        draw = ImageDraw.Draw(img)
        ax, ay = ankle
        ox = size * 0.55
        draw.ellipse(
            [ax - size * 0.3 + ox, ay - size * 0.25, ax + size * 0.9, ay + size * 0.35],
            fill=fill,
        )

    def _disk(self, img, center, radius, fill) -> None:
        draw = ImageDraw.Draw(img)
        cx, cy = center
        r = max(1.0, radius)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    def _blit_centered(self, canvas, img, cx, cy, scale) -> None:
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)
        if local.mode != "RGBA":
            local = local.convert("RGBA")
        canvas.alpha_composite(local, (int(cx - local.width / 2), int(cy - local.height / 2)))

    def _paste_tile(self, canvas, img, ox, oy, tile_x) -> None:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        if not tile_x:
            canvas.alpha_composite(img, (ox, oy))
            return
        iw = img.size[0]
        if iw <= 0:
            return
        cw = canvas.size[0]
        x = (ox % iw) - iw
        while x < cw + iw:
            canvas.alpha_composite(img, (x, oy))
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
