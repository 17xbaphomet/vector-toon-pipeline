"""Pure procedural animation helpers (no I/O)."""

from __future__ import annotations

import math
from typing import Sequence

from .entities import VisemeCue
from .value_objects import Affine, CameraState, Viseme


def head_bob(t: float, amplitude: float = 4.0, freq: float = 2.5) -> float:
    return amplitude * math.sin(2 * math.pi * freq * t)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ease_smooth(u: float) -> float:
    u = _clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _two_bone_ik(
    hip: tuple[float, float],
    foot: tuple[float, float],
    thigh_len: float,
    shin_len: float,
) -> tuple[float, float, tuple[float, float]]:
    """
    2-bone IK for profile view (y+ down, x+ forward when facing right).

    Returns (thigh_world_deg, shin_world_deg, knee_pos).
    Both angles are ABSOLUTE from downward axis (0 = straight down,
    positive = toward +x). Shin angle points knee → foot.
    Knee is placed on the -x side of the hip→foot line ("behind").
    """
    dx = foot[0] - hip[0]
    dy = foot[1] - hip[1]
    dist = math.hypot(dx, dy)
    max_reach = thigh_len + shin_len - 1.0
    min_reach = abs(thigh_len - shin_len) + 1.0
    dist = _clamp(dist, min_reach, max_reach)

    # Direction hip → foot from downward axis
    target = math.degrees(math.atan2(dx, dy))

    cos_a = (thigh_len**2 + dist**2 - shin_len**2) / (2 * thigh_len * dist)
    a = math.degrees(math.acos(_clamp(cos_a, -1.0, 1.0)))

    # Put knee BEHIND the leg (toward -x relative to target direction)
    # target - a swings thigh so knee is posterior for typical upright pose
    thigh_ang = target - a

    rad = math.radians(thigh_ang)
    knee = (
        hip[0] + math.sin(rad) * thigh_len,
        hip[1] + math.cos(rad) * thigh_len,
    )

    # Absolute shin angle: knee → foot
    sdx = foot[0] - knee[0]
    sdy = foot[1] - knee[1]
    shin_ang = math.degrees(math.atan2(sdx, sdy))

    return thigh_ang, shin_ang, knee


def grounded_walk(
    t: float,
    *,
    step_length: float = 55.0,
    cycle: float = 0.6,
    thigh_len: float = 50.0,
    shin_len: float = 48.0,
    hip_height: float | None = None,
    bob_amp: float = 4.0,
    facing: float = 1.0,
) -> dict:
    """
    Grounded walk – stance foot world-locked, swing foot arcs forward.

    Bone angles are ABSOLUTE world angles (see _two_bone_ik).
    scroll_speed = step_length / (cycle/2) so bg can match body velocity.
    """
    if hip_height is None:
        hip_height = thigh_len + shin_len  # ~98: straight-leg hip above ground

    step_period = cycle / 2.0
    scroll_speed = step_length / step_period
    body_world_x = scroll_speed * t

    total_steps = body_world_x / step_length
    step_i = int(math.floor(total_steps))
    u = total_steps - step_i

    if step_i % 2 == 0:
        # Left stance, right swings
        stance_plant = step_i * step_length
        swing_from = (step_i - 1) * step_length
        swing_to = (step_i + 1) * step_length
        stance_is_left = True
    else:
        stance_plant = step_i * step_length
        swing_from = (step_i - 1) * step_length
        swing_to = (step_i + 1) * step_length
        stance_is_left = False

    su = _ease_smooth(u)
    swing_wx = swing_from + (swing_to - swing_from) * su
    swing_lift = 16.0 * math.sin(math.pi * u)  # lift above ground (y- up in local)

    if stance_is_left:
        left_wx, left_wy = stance_plant, 0.0
        right_wx, right_wy = swing_wx, -swing_lift
    else:
        right_wx, right_wy = stance_plant, 0.0
        left_wx, left_wy = swing_wx, -swing_lift

    bob = bob_amp * abs(math.sin(math.pi * total_steps))

    # Body-local coords: origin under body on ground, y+ down
    def to_local(wx: float, wy: float) -> tuple[float, float]:
        return ((wx - body_world_x) * facing, wy)

    left_foot = to_local(left_wx, left_wy)
    right_foot = to_local(right_wx, right_wy)

    # Hip fixed under body center (screen-centered character)
    hip = (0.0, -(hip_height - bob))

    l_th, l_sh, _ = _two_bone_ik(hip, left_foot, thigh_len, shin_len)
    r_th, r_sh, _ = _two_bone_ik(hip, right_foot, thigh_len, shin_len)

    # Arms opposite to legs (absolute angles from down)
    l_ua = -0.55 * l_th
    r_ua = -0.55 * r_th
    # Elbow: slight bend, absolute ≈ upper + relative bend
    l_fa = l_ua - 35.0
    r_fa = r_ua - 35.0

    bones: dict[str, Affine] = {
        "body": Affine.translate(0.0, bob),
        "head": Affine.rotate(2.0 * math.sin(2 * math.pi * t / cycle)),
        # ABSOLUTE angles for thigh AND shin
        "left_thigh": Affine.rotate(l_th),
        "left_shin": Affine.rotate(l_sh),
        "right_thigh": Affine.rotate(r_th),
        "right_shin": Affine.rotate(r_sh),
        "left_upper_arm": Affine.rotate(l_ua),
        "left_forearm": Affine.rotate(l_fa),
        "right_upper_arm": Affine.rotate(r_ua),
        "right_forearm": Affine.rotate(r_fa),
        "left_leg": Affine.rotate(l_th),
        "right_leg": Affine.rotate(r_th),
        "left_arm": Affine.rotate(l_ua),
        "right_arm": Affine.rotate(r_ua),
    }

    return {
        "body_world_x": body_world_x,
        "scroll_speed": scroll_speed,
        "hip_local": hip,
        "bones": bones,
        "feet_local": {"left": left_foot, "right": right_foot},
        "hip_height": hip_height,
        "step_length": step_length,
        "cycle": cycle,
    }


def walk_cycle_pose(
    t: float,
    stride: float = 28.0,
    bob_amp: float = 5.0,
    cycle_duration: float = 0.55,
) -> dict[str, Affine]:
    return grounded_walk(
        t, step_length=stride * 2.0, cycle=cycle_duration, bob_amp=bob_amp
    )["bones"]


def path_position(
    t: float,
    path: Sequence[tuple[float, float]],
    speed: float = 80.0,
) -> tuple[tuple[float, float], float, float]:
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

    dist = min(dist_raw, total - 1e-6)
    traveled = 0.0
    for a, b, length in segs:
        if traveled + length >= dist:
            uu = (dist - traveled) / length if length > 0 else 0.0
            x = a[0] + (b[0] - a[0]) * uu
            y = a[1] + (b[1] - a[1]) * uu
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
    t: float, actions: Sequence, default: CameraState | None = None
) -> CameraState:
    default = default or CameraState()
    for act in actions:
        if act.type not in ("camera", "pan") or not act.timing.contains(t):
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
    return (
        -camera.x * layer_parallax + scroll_x * t,
        -camera.y * layer_parallax + scroll_y * t,
    )


def facing_from_heading(heading_deg: float) -> float:
    return 1.0 if math.cos(math.radians(heading_deg)) >= 0 else -1.0
