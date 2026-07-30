"""Pure procedural animation helpers (no I/O)."""

from __future__ import annotations

import math
from typing import Sequence

from .entities import VisemeCue
from .value_objects import Affine, CameraState, Viseme


def head_bob(t: float, amplitude: float = 4.0, freq: float = 2.5) -> float:
    return amplitude * math.sin(2 * math.pi * freq * t)


def walk_cycle_pose(
    t: float,
    stride: float = 28.0,
    bob_amp: float = 5.0,
    cycle_duration: float = 0.55,
) -> dict[str, Affine]:
    """
    Realistic-ish bipedal walk for *profile* view.

    Returns rotations (degrees) encoded as Affine.rotate around limb pivots.
    Bones:
      body, head,
      left_thigh, left_shin, right_thigh, right_shin,
      left_upper_arm, left_forearm, right_upper_arm, right_forearm

    Phase 0 = left foot contact, right foot swinging forward.
    """
    phase = (t % cycle_duration) / cycle_duration  # 0..1
    a = phase * 2 * math.pi

    # --- Legs (degrees) ---
    # Hip swing ~±30°, knee bends more on swing leg (pass-through)
    hip_L = stride * math.sin(a)                    # + forward
    hip_R = stride * math.sin(a + math.pi)

    # Knee: mostly bent during swing (when hip is forward-ish after push-off)
    # Classic: max flexion mid-swing
    knee_L = max(0.0, -55.0 * max(0.0, math.sin(a + 0.3)))   # bend when swinging
    knee_R = max(0.0, -55.0 * max(0.0, math.sin(a + math.pi + 0.3)))
    # stance leg slight flex
    knee_L += -8.0 * max(0.0, -math.sin(a))
    knee_R += -8.0 * max(0.0, -math.sin(a + math.pi))

    # --- Arms (opposite to legs) ---
    sh_L = -0.75 * hip_L   # opposite phase
    sh_R = -0.75 * hip_R
    # Elbow always slightly bent, more on forward swing
    el_L = -25.0 - 20.0 * max(0.0, math.sin(a + math.pi))
    el_R = -25.0 - 20.0 * max(0.0, math.sin(a))

    body_bob = bob_amp * abs(math.sin(a))
    # slight body lean into walk direction handled by facing, not here

    return {
        "body": Affine.translate(0.0, -body_bob),
        "head": Affine.rotate(3.0 * math.sin(a), cx=0, cy=0),
        "left_thigh": Affine.rotate(hip_L),
        "left_shin": Affine.rotate(knee_L),
        "right_thigh": Affine.rotate(hip_R),
        "right_shin": Affine.rotate(knee_R),
        "left_upper_arm": Affine.rotate(sh_L),
        "left_forearm": Affine.rotate(el_L),
        "right_upper_arm": Affine.rotate(sh_R),
        "right_forearm": Affine.rotate(el_R),
        # aliases for simpler rigs
        "left_leg": Affine.rotate(hip_L),
        "right_leg": Affine.rotate(hip_R),
        "left_arm": Affine.rotate(sh_L),
        "right_arm": Affine.rotate(sh_R),
    }


def path_position(
    t: float,
    path: Sequence[tuple[float, float]],
    speed: float = 80.0,
) -> tuple[tuple[float, float], float, float]:
    """
    Follow polyline at constant speed.
    Returns ((x, y), heading_degrees, distance_traveled).
    distance_traveled is absolute along path (not modulo) for scroll.
    """
    if not path or len(path) < 2:
        p0 = path[0] if path else (0.0, 0.0)
        return p0, 0.0, t * speed

    segs: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        segs.append((a, b, length))
        total += length

    dist_raw = t * speed
    if total <= 0:
        return path[0], 0.0, dist_raw

    # Clamp to end (no loop) for continuous one-direction walks
    dist = min(dist_raw, total - 1e-6)
    traveled = 0.0
    for a, b, length in segs:
        if traveled + length >= dist:
            u = (dist - traveled) / length if length > 0 else 0.0
            x = a[0] + (b[0] - a[0]) * u
            y = a[1] + (b[1] - a[1]) * u
            heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            return (x, y), heading, dist_raw
        traveled += length

    return path[-1], 0.0, dist_raw


def sample_viseme_at(
    cues: Sequence[VisemeCue], t: float
) -> tuple[Viseme, float]:
    for cue in cues:
        if cue.timing.contains(t):
            return cue.value, cue.intensity
    return Viseme.X, 0.0


def camera_at(
    t: float,
    actions: Sequence,
    default: CameraState | None = None,
) -> CameraState:
    default = default or CameraState()
    for act in actions:
        if act.type not in ("camera", "pan"):
            continue
        if not act.timing.contains(t):
            continue
        params = act.params
        u = (t - act.timing.start) / max(act.timing.duration, 1e-6)
        fx, fy = params.get("from", (default.x, default.y))
        tx, ty = params.get("to", (fx, fy))
        z0 = float(params.get("zoom_from", default.zoom))
        z1 = float(params.get("zoom_to", z0))
        return CameraState(
            x=fx + (tx - fx) * u,
            y=fy + (ty - fy) * u,
            zoom=z0 + (z1 - z0) * u,
        )
    return default


def parallax_offset(
    camera: CameraState,
    layer_parallax: float,
    t: float,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
) -> tuple[float, float]:
    """World offset: camera + continuous scroll (scroll already encodes direction)."""
    return (
        -camera.x * layer_parallax + scroll_x * t,
        -camera.y * layer_parallax + scroll_y * t,
    )


def facing_from_heading(heading_deg: float) -> float:
    """+1 = facing right (profile), -1 = facing left."""
    # heading 0 = right, 180 = left
    cos_h = math.cos(math.radians(heading_deg))
    return 1.0 if cos_h >= 0 else -1.0
