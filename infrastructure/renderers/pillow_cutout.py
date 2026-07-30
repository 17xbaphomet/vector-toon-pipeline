"""Frame renderer with rigidly anchored joint chains."""

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


def _ang(tf: Affine) -> float:
    return math.degrees(math.atan2(tf.b, tf.a))


def _offset(angle_deg: float, length: float) -> tuple[float, float]:
    """Step from a joint along angle (0=down, + = toward +x)."""
    rad = math.radians(angle_deg)
    return math.sin(rad) * length, math.cos(rad) * length


class PillowCutoutRenderer(FrameRenderer):
    """
    Rigid kinematic chains:

        hip ──thigh_ang, L──► knee ──shin_ang, L──► ankle
        shoulder ──ua_ang, L──► elbow ──fa_ang, L──► hand

    Every distal joint is computed ONLY from its parent joint.
    Segment lengths are the single source of truth (must match IK).
    """

    # Must match domain/procedural.grounded_walk defaults
    THIGH = 50.0
    SHIN = 48.0
    UPPER = 40.0
    FORE = 38.0
    HIP_H = 98.0  # THIGH + SHIN

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

        cx, cy = state.root_position  # cy ≈ ground under character
        flip = state.scale < 0
        s = abs(state.scale) if state.scale else 1.0
        bones = state.bone_transforms

        bob = bones.get("body", Affine.identity()).f * s

        # ── Skeleton root anchors (screen space) ──────────────────────
        # Ground at cy; hip sits HIP_H above ground.
        hip_y = cy - self.HIP_H * s + bob
        hip_x = cx
        # Torso center sits a bit above hip (ellipse mid)
        torso_x, torso_y = cx, hip_y - 35 * s
        # Shoulders above hip along torso
        sh_y = hip_y - 70 * s
        sh_x = cx

        side = Path(rig.base_svg).parent / "body_side.svg"
        torso_path = side if side.is_file() else rig.base_svg
        limbs = Path(rig.base_svg).parent / "limbs"

        torso = self._svg_to_pil(torso_path)
        if flip:
            torso = torso.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # near/far labels for depth
        if not flip:
            far = "right"
            near = "left"
            far_dx, near_dx = -4.0, 4.0
        else:
            far = "left"
            near = "right"
            far_dx, near_dx = -4.0, 4.0

        # 1) FAR limbs (behind torso)
        self._chain_leg(
            canvas, limbs, bones, flip, s,
            hip=(hip_x + far_dx * s, hip_y),
            side=far,
        )
        self._chain_arm(
            canvas, limbs, bones, flip, s,
            shoulder=(sh_x + far_dx * s, sh_y),
            side=far,
        )

        # 2) Torso (anchors stay fixed; limbs attach to same coords)
        self._blit_centered(canvas, torso, torso_x, torso_y, s)

        # 3) NEAR limbs (in front)
        self._chain_leg(
            canvas, limbs, bones, flip, s,
            hip=(hip_x + near_dx * s, hip_y),
            side=near,
        )
        self._chain_arm(
            canvas, limbs, bones, flip, s,
            shoulder=(sh_x + near_dx * s, sh_y),
            side=near,
        )

        # 4) Mouth
        mouth_path = bones and (
            rig.mouth_shapes.get(state.viseme) or rig.mouth_shapes.get(Viseme.X)
        )
        if mouth_path:
            mouth = self._svg_to_pil(mouth_path)
            if flip:
                mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            head_bob = bones.get("head", Affine.identity()).f * s
            mx = cx + (18 if not flip else -18) * s
            my = sh_y - 30 * s + head_bob
            self._blit_centered(canvas, mouth, mx, my, s * 0.65)

        out = self.cache_dir / f"frame_{state.time:.4f}.png"
        canvas.convert("RGB").save(out, "PNG")
        return out

    # ── Rigid chains ─────────────────────────────────────────────────

    def _chain_leg(
        self, canvas, limbs, bones, flip, s,
        hip: tuple[float, float], side: str,
    ) -> None:
        """hip → knee → ankle. Knee is EXACTLY hip + offset(thigh)."""
        thigh_img = self._img(limbs / "thigh.svg")
        shin_img = self._img(limbs / "shin.svg")
        if thigh_img is None:
            return

        th = _ang(bones.get(f"{side}_thigh") or bones.get(f"{side}_leg", Affine.identity()))
        sh = _ang(bones.get(f"{side}_shin", Affine.identity()))
        if flip:
            th, sh = -th, -sh

        hx, hy = hip
        # 1) thigh from hip
        self._blit_limb(canvas, thigh_img, th, hx, hy, s)

        # 2) knee = hip + direction * THIGH  (rigid)
        kx, ky = hx + _offset(th, self.THIGH * s)[0], hy + _offset(th, self.THIGH * s)[1]

        # 3) shin from knee (same point the thigh ends at)
        if shin_img is not None:
            self._blit_limb(canvas, shin_img, sh, kx, ky, s)

    def _chain_arm(
        self, canvas, limbs, bones, flip, s,
        shoulder: tuple[float, float], side: str,
    ) -> None:
        """shoulder → elbow → hand. Elbow = shoulder + offset(upper)."""
        upper_img = self._img(limbs / "upper_arm.svg")
        fore_img = self._img(limbs / "forearm.svg")
        if upper_img is None:
            return

        ua = _ang(bones.get(f"{side}_upper_arm") or bones.get(f"{side}_arm", Affine.identity()))
        fa = _ang(bones.get(f"{side}_forearm", Affine.identity()))
        if flip:
            ua, fa = -ua, -fa

        sx, sy = shoulder
        self._blit_limb(canvas, upper_img, ua, sx, sy, s)

        ex, ey = sx + _offset(ua, self.UPPER * s)[0], sy + _offset(ua, self.UPPER * s)[1]

        if fore_img is not None:
            self._blit_limb(canvas, fore_img, fa, ex, ey, s)

    # ── Blit helpers ─────────────────────────────────────────────────

    def _blit_limb(
        self,
        canvas: Image.Image,
        img: Image.Image,
        angle_deg: float,
        joint_x: float,
        joint_y: float,
        scale: float,
    ) -> None:
        """
        Paste limb so its TOP-CENTER pivot sits exactly on (joint_x, joint_y).
        Rotation around that pivot; length along the sprite is visual only —
        kinematic length is the constant used in _offset().
        """
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)

        pw, ph = local.size
        # pivot = top center of sprite
        piv_x, piv_y = pw / 2.0, 1.0

        diag = int(math.ceil(math.hypot(pw, ph))) + 4
        side = diag * 2
        pad = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        pad.paste(local, (int(diag - piv_x), int(diag - piv_y)), local)

        # PIL rotate is counter-clockwise for positive degrees;
        # our angle is clockwise-from-down in screen y-down space → negate
        rot = pad.rotate(-angle_deg, resample=Image.Resampling.BICUBIC)
        canvas.paste(rot, (int(joint_x - diag), int(joint_y - diag)), rot)

    def _blit_centered(
        self, canvas: Image.Image, img: Image.Image,
        cx: float, cy: float, scale: float,
    ) -> None:
        local = img
        if scale != 1.0:
            nw = max(1, int(local.width * scale))
            nh = max(1, int(local.height * scale))
            local = local.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.paste(
            local,
            (int(cx - local.width / 2), int(cy - local.height / 2)),
            local if local.mode == "RGBA" else None,
        )

    def _img(self, path: Path) -> Image.Image | None:
        if not path.is_file():
            return None
        return self._svg_to_pil(path)

    def _paste_tile(
        self, canvas: Image.Image, img: Image.Image,
        ox: int, oy: int, tile_x: bool,
    ) -> None:
        if not tile_x:
            canvas.paste(img, (ox, oy), img if img.mode == "RGBA" else None)
            return
        iw = img.size[0]
        if iw <= 0:
            return
        cw = canvas.size[0]
        x = (ox % iw) - iw
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
