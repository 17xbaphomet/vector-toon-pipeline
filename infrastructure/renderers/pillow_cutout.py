"""Frame renderer: parallax bg + articulated profile character + mouth."""

from __future__ import annotations

import math
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
    return math.degrees(math.atan2(tf.b, tf.a))


class PillowCutoutRenderer(FrameRenderer):
    """
    Profile walk renderer with correct joint chaining.

    Coordinate convention (root = feet / screen position):
      shoulder ≈ (cx, cy - 125)
      hip      ≈ (cx, cy - 55)
    Limbs hang downward; angle 0 = straight down, positive = swing forward (right).
    """

    # Unscaled segment lengths (match limb SVG heights roughly)
    THIGH_LEN = 50.0
    SHIN_LEN = 48.0
    UPPER_ARM_LEN = 40.0
    FOREARM_LEN = 38.0

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

        cx, cy = state.root_position
        flip = state.scale < 0
        scale = abs(state.scale) if state.scale != 0 else 1.0

        body_tf = state.bone_transforms.get("body", Affine.identity())
        bob_y = body_tf.f * scale

        # Shared skeleton anchors (relative to root / feet)
        hip_y = cy - 55 * scale + bob_y
        shoulder_y = cy - 125 * scale + bob_y
        torso_cy = cy - 95 * scale + bob_y

        side = Path(rig.base_svg).parent / "body_side.svg"
        torso_path = side if side.is_file() else rig.base_svg
        limbs_dir = Path(rig.base_svg).parent / "limbs"

        torso = self._svg_to_pil(torso_path)
        if flip:
            torso = torso.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        self._paste_centered(canvas, torso, cx, torso_cy, scale)

        # Draw order: far limbs first, near limbs on top
        # facing right: right = far, left = near
        self._draw_leg_chain(
            canvas, limbs_dir, state,
            hip_x=cx + (-6 if not flip else 6) * scale,
            hip_y=hip_y,
            scale=scale, flip=flip,
            thigh_key="right_thigh", shin_key="right_shin",
        )
        self._draw_arm_chain(
            canvas, limbs_dir, state,
            sh_x=cx + (-4 if not flip else 4) * scale,
            sh_y=shoulder_y,
            scale=scale, flip=flip,
            upper_key="right_upper_arm", fore_key="right_forearm",
        )
        self._draw_leg_chain(
            canvas, limbs_dir, state,
            hip_x=cx + (6 if not flip else -6) * scale,
            hip_y=hip_y,
            scale=scale, flip=flip,
            thigh_key="left_thigh", shin_key="left_shin",
        )
        self._draw_arm_chain(
            canvas, limbs_dir, state,
            sh_x=cx + (4 if not flip else -4) * scale,
            sh_y=shoulder_y,
            scale=scale, flip=flip,
            upper_key="left_upper_arm", fore_key="left_forearm",
        )

        mouth_path = rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        if mouth_path:
            mouth = self._svg_to_pil(mouth_path)
            if flip:
                mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            head_tf = state.bone_transforms.get("head", Affine.identity())
            mx = cx + (22 if not flip else -22) * scale
            my = cy - 155 * scale + bob_y + head_tf.f * scale
            self._paste_centered(canvas, mouth, mx, my, scale * 0.65)

        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        canvas.convert("RGB").save(out, "PNG")
        return out

    # ------------------------------------------------------------------
    # Limb chains
    # ------------------------------------------------------------------

    def _draw_leg_chain(
        self, canvas, limbs_dir, state,
        hip_x, hip_y, scale, flip, thigh_key, shin_key,
    ) -> None:
        thigh = self._limb_img(limbs_dir / "thigh.svg", flip=False)
        shin = self._limb_img(limbs_dir / "shin.svg", flip=False)
        if thigh is None:
            return

        th_tf = state.bone_transforms.get(thigh_key) or state.bone_transforms.get(
            "left_leg" if "left" in thigh_key else "right_leg", Affine.identity()
        )
        sh_tf = state.bone_transforms.get(shin_key, Affine.identity())

        hip_ang = _rotation_deg(th_tf)
        knee_ang = _rotation_deg(sh_tf)  # local relative bend
        if flip:
            hip_ang = -hip_ang
            knee_ang = -knee_ang

        # Angle 0 = straight down; positive swings toward +x (forward when facing right)
        self._paste_limb_at_joint(canvas, thigh, hip_ang, hip_x, hip_y, scale)

        thigh_len = self.THIGH_LEN * scale
        rad = math.radians(hip_ang)
        knee_x = hip_x + math.sin(rad) * thigh_len
        knee_y = hip_y + math.cos(rad) * thigh_len

        if shin is not None:
            world_shin = hip_ang + knee_ang
            self._paste_limb_at_joint(canvas, shin, world_shin, knee_x, knee_y, scale)

    def _draw_arm_chain(
        self, canvas, limbs_dir, state,
        sh_x, sh_y, scale, flip, upper_key, fore_key,
    ) -> None:
        upper = self._limb_img(limbs_dir / "upper_arm.svg", flip=False)
        fore = self._limb_img(limbs_dir / "forearm.svg", flip=False)
        if upper is None:
            return

        u_tf = state.bone_transforms.get(upper_key) or state.bone_transforms.get(
            "left_arm" if "left" in upper_key else "right_arm", Affine.identity()
        )
        f_tf = state.bone_transforms.get(fore_key, Affine.identity())

        sh_ang = _rotation_deg(u_tf)
        el_ang = _rotation_deg(f_tf)
        if flip:
            sh_ang = -sh_ang
            el_ang = -el_ang

        self._paste_limb_at_joint(canvas, upper, sh_ang, sh_x, sh_y, scale)

        arm_len = self.UPPER_ARM_LEN * scale
        rad = math.radians(sh_ang)
        elbow_x = sh_x + math.sin(rad) * arm_len
        elbow_y = sh_y + math.cos(rad) * arm_len

        if fore is not None:
            world_fore = sh_ang + el_ang
            self._paste_limb_at_joint(canvas, fore, world_fore, elbow_x, elbow_y, scale)

    # ------------------------------------------------------------------
    # Pivot-stable paste
    # ------------------------------------------------------------------

    def _paste_limb_at_joint(
        self,
        canvas: Image.Image,
        img: Image.Image,
        angle_deg: float,
        joint_x: float,
        joint_y: float,
        scale: float,
    ) -> None:
        """
        Scale limb, rotate around its top-center pivot, paste so that
        pivot lands exactly on (joint_x, joint_y).
        """
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)

        w, h = local.size
        # Pivot: top-center of the limb sprite
        px, py = w / 2.0, 2.0

        # Pad so pivot sits at the center of a square large enough to rotate into
        diag = int(math.ceil(math.hypot(w, h))) + 2
        side = diag * 2
        pad = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        origin = (int(diag - px), int(diag - py))
        pad.paste(local, origin, local)

        rotated = pad.rotate(-angle_deg, resample=Image.Resampling.BICUBIC)
        # After rotation around pad center, pivot is still at (diag, diag)
        dest = (int(joint_x - diag), int(joint_y - diag))
        canvas.paste(rotated, dest, rotated)

    def _paste_centered(
        self,
        canvas: Image.Image,
        img: Image.Image,
        cx: float,
        cy: float,
        scale: float,
    ) -> None:
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)
        x = int(cx - local.width / 2)
        y = int(cy - local.height / 2)
        canvas.paste(local, (x, y), local if local.mode == "RGBA" else None)

    def _limb_img(self, path: Path, flip: bool = False) -> Image.Image | None:
        if not path.is_file():
            return None
        img = self._svg_to_pil(path)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img

    def _paste(
        self,
        canvas: Image.Image,
        img: Image.Image,
        ox: int,
        oy: int,
        tile_x: bool = False,
    ) -> None:
        cw, _ = canvas.size
        iw = img.size[0]
        if not tile_x:
            canvas.paste(img, (ox, oy), img if img.mode == "RGBA" else None)
            return
        if iw <= 0:
            return
        start = (ox % iw) - iw
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
