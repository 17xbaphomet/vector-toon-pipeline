"""Frame renderer: parallax bg + articulated profile character + mouth."""

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


def _rotation_deg(tf: Affine) -> float:
    """Extract approximate rotation from affine (from a,b)."""
    import math

    return math.degrees(math.atan2(tf.b, tf.a))


class PillowCutoutRenderer(FrameRenderer):
    """
    Profile walk renderer:
    - parallax / scrolling backgrounds
    - torso (body_side) + articulated limbs (thigh→shin, upper_arm→forearm)
    - horizontal flip when scale < 0 (facing left)
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
                    state.camera,
                    layer.parallax,
                    state.time,
                    layer.scroll_x,
                    layer.scroll_y,
                )
                self._paste(canvas, img, int(ox), int(oy), tile_x=layer.repeat_x)

        # Character root
        cx, cy = state.root_position
        flip = state.scale < 0
        scale = abs(state.scale) if state.scale != 0 else 1.0

        # Prefer side view if available
        side = Path(rig.base_svg).parent / "body_side.svg"
        torso_path = side if side.is_file() else rig.base_svg
        limbs_dir = Path(rig.base_svg).parent / "limbs"

        torso = self._svg_to_pil(torso_path)
        if flip:
            torso = torso.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        body_tf = state.bone_transforms.get("body", Affine.identity())
        bob_y = body_tf.f

        self._paste_scaled(canvas, torso, cx, cy - 40 + bob_y, scale)

        # Limb chain: back limbs first, then front
        # Profile facing right: right limbs slightly behind
        self._draw_leg(
            canvas, limbs_dir, state, cx, cy, scale, flip,
            thigh_key="right_thigh", shin_key="right_shin",
            hip_off=(-8, 10), behind=True,
        )
        self._draw_arm(
            canvas, limbs_dir, state, cx, cy, scale, flip,
            upper_key="right_upper_arm", fore_key="right_forearm",
            shoulder_off=(-6, -35), behind=True,
        )
        self._draw_leg(
            canvas, limbs_dir, state, cx, cy, scale, flip,
            thigh_key="left_thigh", shin_key="left_shin",
            hip_off=(6, 10), behind=False,
        )
        self._draw_arm(
            canvas, limbs_dir, state, cx, cy, scale, flip,
            upper_key="left_upper_arm", fore_key="left_forearm",
            shoulder_off=(8, -35), behind=False,
        )

        # Mouth on head region
        mouth_path = rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        if mouth_path:
            mouth = self._svg_to_pil(mouth_path)
            if flip:
                mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            head_tf = state.bone_transforms.get("head", Affine.identity())
            mx = cx + (18 if not flip else -18) * scale
            my = cy - 95 * scale + head_tf.f + bob_y
            self._paste_scaled(canvas, mouth, mx, my, scale * 0.7)

        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        canvas.convert("RGB").save(out, "PNG")
        return out

    def _draw_leg(
        self, canvas, limbs_dir, state, cx, cy, scale, flip,
        thigh_key, shin_key, hip_off, behind,
    ):
        thigh = self._limb_img(limbs_dir / "thigh.svg", flip)
        shin = self._limb_img(limbs_dir / "shin.svg", flip)
        if thigh is None:
            return

        th_tf = state.bone_transforms.get(thigh_key) or state.bone_transforms.get(
            "left_leg" if "left" in thigh_key else "right_leg", Affine.identity()
        )
        sh_tf = state.bone_transforms.get(shin_key, Affine.identity())
        hip_ang = _rotation_deg(th_tf)
        knee_ang = _rotation_deg(sh_tf)

        ox, oy = hip_off
        if flip:
            ox = -ox
        hx = cx + ox * scale
        hy = cy + oy * scale

        thigh_r = self._rotate(thigh, hip_ang if not flip else -hip_ang)
        self._paste_scaled(canvas, thigh_r, hx, hy + 20 * scale, scale, pivot="top")

        # Knee position approx: hip + rotated thigh length
        import math

        thigh_len = 48 * scale
        ang = math.radians(hip_ang if not flip else -hip_ang)
        kx = hx + math.sin(ang) * thigh_len
        ky = hy + math.cos(ang) * thigh_len
        if shin is not None:
            shin_r = self._rotate(shin, (hip_ang + knee_ang) if not flip else -(hip_ang + knee_ang))
            self._paste_scaled(canvas, shin_r, kx, ky + 18 * scale, scale, pivot="top")

    def _draw_arm(
        self, canvas, limbs_dir, state, cx, cy, scale, flip,
        upper_key, fore_key, shoulder_off, behind,
    ):
        upper = self._limb_img(limbs_dir / "upper_arm.svg", flip)
        fore = self._limb_img(limbs_dir / "forearm.svg", flip)
        if upper is None:
            return

        u_tf = state.bone_transforms.get(upper_key) or state.bone_transforms.get(
            "left_arm" if "left" in upper_key else "right_arm", Affine.identity()
        )
        f_tf = state.bone_transforms.get(fore_key, Affine.identity())
        sh_ang = _rotation_deg(u_tf)
        el_ang = _rotation_deg(f_tf)

        ox, oy = shoulder_off
        if flip:
            ox = -ox
        sx = cx + ox * scale
        sy = cy + oy * scale

        upper_r = self._rotate(upper, sh_ang if not flip else -sh_ang)
        self._paste_scaled(canvas, upper_r, sx, sy + 15 * scale, scale, pivot="top")

        import math

        arm_len = 38 * scale
        ang = math.radians(sh_ang if not flip else -sh_ang)
        ex = sx + math.sin(ang) * arm_len
        ey = sy + math.cos(ang) * arm_len
        if fore is not None:
            fore_r = self._rotate(fore, (sh_ang + el_ang) if not flip else -(sh_ang + el_ang))
            self._paste_scaled(canvas, fore_r, ex, ey + 14 * scale, scale, pivot="top")

    def _limb_img(self, path: Path, flip: bool) -> Image.Image | None:
        if not path.is_file():
            return None
        img = self._svg_to_pil(path)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img

    def _rotate(self, img: Image.Image, degrees: float) -> Image.Image:
        return img.rotate(-degrees, expand=True, resample=Image.Resampling.BICUBIC)

    def _paste_scaled(
        self,
        canvas: Image.Image,
        img: Image.Image,
        cx: float,
        cy: float,
        scale: float,
        pivot: str = "center",
    ) -> None:
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)
        if pivot == "top":
            x = int(cx - local.width / 2)
            y = int(cy)
        else:
            x = int(cx - local.width / 2)
            y = int(cy - local.height / 2)
        canvas.paste(local, (x, y), local if local.mode == "RGBA" else None)

    def _paste(
        self,
        canvas: Image.Image,
        img: Image.Image,
        ox: int,
        oy: int,
        tile_x: bool = False,
    ) -> None:
        cw, ch = canvas.size
        iw, ih = img.size
        if not tile_x:
            canvas.paste(img, (ox, oy), img if img.mode == "RGBA" else None)
            return
        start = (ox % iw) - iw if iw else 0
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
