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

# Arm lengths (shared with procedural if exported; keep local fallback)
UPPER_ARM_LEN = 40.0
FOREARM_LEN = 38.0


def _ang(tf: Affine) -> float:
    return math.degrees(math.atan2(tf.b, tf.a))


def _dir(angle_deg: float, length: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return math.sin(rad) * length, math.cos(rad) * length


class PillowCutoutRenderer(FrameRenderer):
    """
    Torso from SVG; limbs drawn as capsules between joints.

    Joints are shared by construction:
      knee  = hip + dir(thigh_angle) * THIGH_LEN
      ankle = knee + dir(shin_angle) * SHIN_LEN
    Both thigh and shin capsules end/start on the exact same knee pixel.
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

        # --- skeleton anchors (screen) ---
        hip_x, hip_y = cx, cy - HIP_HEIGHT * s + bob
        sh_x, sh_y = cx, hip_y - 70 * s
        torso_x, torso_y = cx, hip_y - 30 * s

        side_svg = Path(rig.base_svg).parent / "body_side.svg"
        torso_path = side_svg if side_svg.is_file() else rig.base_svg
        torso = self._svg_to_pil(torso_path)
        if flip:
            torso = torso.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Colors
        leg_col = (44, 95, 138, 255)       # #2C5F8A
        leg_outline = (26, 58, 85, 255)
        arm_col = (245, 203, 167, 255)     # #F5CBA7
        arm_outline = (212, 165, 116, 255)
        foot_col = (26, 26, 26, 255)

        if not flip:
            order = ["right", "left"]   # far then near
            dx = {"right": -5.0, "left": 5.0}
        else:
            order = ["left", "right"]
            dx = {"left": -5.0, "right": 5.0}

        # Layer buffer so we can composite far → torso → near
        far_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        near_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        for i, side in enumerate(order):
            layer = far_layer if i == 0 else near_layer
            hx = hip_x + dx[side] * s
            sx = sh_x + dx[side] * s

            # --- Leg angles ---
            th = _ang(bones.get(f"{side}_thigh") or bones.get(f"{side}_leg", Affine.identity()))
            shn = _ang(bones.get(f"{side}_shin", Affine.identity()))
            if flip:
                th, shn = -th, -shn

            # Rigid FK joints
            knee = (hx + _dir(th, THIGH_LEN * s)[0], hip_y + _dir(th, THIGH_LEN * s)[1])
            ankle = (knee[0] + _dir(shn, SHIN_LEN * s)[0], knee[1] + _dir(shn, SHIN_LEN * s)[1])

            self._capsule(layer, (hx, hip_y), knee, 9 * s, leg_col, leg_outline)
            self._capsule(layer, knee, ankle, 7 * s, leg_col, leg_outline)
            # foot ellipse at ankle
            self._foot(layer, ankle, shn, 12 * s, foot_col)

            # --- Arm angles ---
            ua = _ang(bones.get(f"{side}_upper_arm") or bones.get(f"{side}_arm", Affine.identity()))
            fa = _ang(bones.get(f"{side}_forearm", Affine.identity()))
            if flip:
                ua, fa = -ua, -fa

            elbow = (sx + _dir(ua, UPPER_ARM_LEN * s)[0], sh_y + _dir(ua, UPPER_ARM_LEN * s)[1])
            hand = (elbow[0] + _dir(fa, FOREARM_LEN * s)[0], elbow[1] + _dir(fa, FOREARM_LEN * s)[1])

            self._capsule(layer, (sx, sh_y), elbow, 6 * s, arm_col, arm_outline)
            self._capsule(layer, elbow, hand, 5 * s, arm_col, arm_outline)
            # hand blob
            self._disk(layer, hand, 6 * s, arm_col)

        # Composite: far limbs → torso → near limbs
        canvas = Image.alpha_composite(canvas, far_layer)
        self._blit_centered(canvas, torso, torso_x, torso_y, s)
        canvas = Image.alpha_composite(canvas, near_layer)

        # Mouth
        mouth_path = rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        if mouth_path:
            mouth = self._svg_to_pil(mouth_path)
            if flip:
                mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            head_bob = bones.get("head", Affine.identity()).f * s
            mx = cx + (18 if not flip else -18) * s
            my = sh_y - 28 * s + head_bob
            self._blit_centered(canvas, mouth, mx, my, s * 0.65)

        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        canvas.convert("RGB").save(out, "PNG")
        return out

    # ── geometry ─────────────────────────────────────────────────────

    def _capsule(
        self,
        img: Image.Image,
        p0: tuple[float, float],
        p1: tuple[float, float],
        radius: float,
        fill: tuple[int, int, int, int],
        outline: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Filled capsule (stadium) from p0 to p1 — joints share endpoints."""
        draw = ImageDraw.Draw(img)
        x0, y0 = p0
        x1, y1 = p1
        r = max(1.0, radius)

        # Thick line as the shaft
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=int(r * 2))
        # Round caps at both joints (identical points for connected segments)
        for cx_, cy_ in (p0, p1):
            draw.ellipse(
                [cx_ - r, cy_ - r, cx_ + r, cy_ + r],
                fill=fill,
                outline=outline,
            )
        if outline is not None:
            # outline along edges approx via slightly wider stroke underneath already filled
            pass

    def _foot(
        self,
        img: Image.Image,
        ankle: tuple[float, float],
        shin_ang: float,
        size: float,
        fill: tuple[int, int, int, int],
    ) -> None:
        """Small foot ellipse oriented perpendicular-ish to shin."""
        draw = ImageDraw.Draw(img)
        ax, ay = ankle
        # foot extends forward (positive angle direction)
        fx = ax + math.sin(math.radians(shin_ang + 90)) * size * 0.3
        # simpler: just a blob slightly forward of ankle
        forward = math.sin(math.radians(shin_ang))
        # foot along +x when shin is vertical
        ox = size * 0.55
        draw.ellipse(
            [ax - size * 0.3 + ox, ay - size * 0.25,
             ax + size * 0.9, ay + size * 0.35],
            fill=fill,
        )

    def _disk(
        self,
        img: Image.Image,
        center: tuple[float, float],
        radius: float,
        fill: tuple[int, int, int, int],
    ) -> None:
        draw = ImageDraw.Draw(img)
        cx, cy = center
        r = max(1.0, radius)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    def _blit_centered(
        self, canvas: Image.Image, img: Image.Image,
        cx: float, cy: float, scale: float,
    ) -> None:
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)
        x = int(cx - local.width / 2)
        y = int(cy - local.height / 2)
        # paste onto RGBA canvas
        if local.mode != "RGBA":
            local = local.convert("RGBA")
        canvas.alpha_composite(local, (x, y))

    def _paste_tile(
        self, canvas: Image.Image, img: Image.Image,
        ox: int, oy: int, tile_x: bool,
    ) -> None:
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
