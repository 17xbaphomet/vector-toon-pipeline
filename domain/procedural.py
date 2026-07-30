"""Pure procedural animation helpers (no I/O)."""

from __future__ import annotations

import math
from typing import Sequence

from .entities import VisemeCue
from .value_objects import Affine, CameraState, Viseme

THIGH_LEN = 50.0
SHIN_LEN = 48.0
UPPER_ARM_LEN = 40.0
FOREARM_LEN = 38.0
HIP_HEIGHT = 86.0


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
    thigh_len: float = THIGH_LEN,
    shin_len: float = SHIN_LEN,
) -> tuple[float, float, tuple[float, float]]:
    dx = foot[0] - hip[0]
    dy = foot[1] - hip[1]
    dist = math.hypot(dx, dy)
    max_reach = thigh_len + shin_len - 1.0
    min_reach = abs(thigh_len - shin_len) + 1.0
    if dist > max_reach and dist > 1e-6:
        s = max_reach / dist
        foot = (hip[0] + dx * s, hip[1] + dy * s)
        dx, dy = foot[0] - hip[0], foot[1] - hip[1]
        dist = max_reach
    dist = _clamp(dist, min_reach, max_reach)
    target = math.degrees(math.atan2(dx, dy))
    cos_a = (thigh_len ** 2 + dist ** 2 - shin_len ** 2) / (2 * thigh_len * dist)
    a = math.degrees(math.acos(_clamp(cos_a, -1.0, 1.0)))
    thigh_ang = target - a
    rad = math.radians(thigh_ang)
    knee = (hip[0] + math.sin(rad) * thigh_len, hip[1] + math.cos(rad) * thigh_len)
    shin_ang = math.degrees(math.atan2(foot[0] - knee[0], foot[1] - knee[1]))
    return thigh_ang, shin_ang, knee


def grounded_walk(
    t: float,
    *,
    step_length: float = 40.0,
    cycle: float = 0.6,
    thigh_len: float = THIGH_LEN,
    shin_len: float = SHIN_LEN,
    hip_height: float = HIP_HEIGHT,
    bob_amp: float = 3.5,
    facing: float = 1.0,
) -> dict:
    """Grounded walk — IK angles as computed (no post-flip). Movement was good."""
    step_period = cycle / 2.0
    scroll_speed = step_length / step_period
    body_world_x = scroll_speed * t
    total_steps = body_world_x / step_length
    step_i = int(math.floor(total_steps))
    u = total_steps - step_i
    stance_plant = step_i * step_length
    swing_from = (step_i - 1) * step_length
    swing_to = (step_i + 1) * step_length
    stance_is_left = step_i % 2 == 0
    su = _ease_smooth(u)
    swing_wx = swing_from + (swing_to - swing_from) * su
    swing_lift = 14.0 * math.sin(math.pi * u)
    if stance_is_left:
        left_wx, left_wy = stance_plant, 0.0
        right_wx, right_wy = swing_wx, -swing_lift
    else:
        right_wx, right_wy = stance_plant, 0.0
        left_wx, left_wy = swing_wx, -swing_lift
    bob = bob_amp * abs(math.sin(math.pi * total_steps))

    def to_local(wx: float, wy: float) -> tuple[float, float]:
        return ((wx - body_world_x) * facing, wy)

    left_foot = to_local(left_wx, left_wy)
    right_foot = to_local(right_wx, right_wy)
    hip = (0.0, -(hip_height - bob))

    l_th, l_sh, _ = _two_bone_ik(hip, left_foot, thigh_len, shin_len)
    r_th, r_sh, _ = _two_bone_ik(hip, right_foot, thigh_len, shin_len)

    # Arms: opposite phase to legs (natural walk)
    # When left leg forward (+), left arm goes back (-)
    l_ua = -0.65 * l_th
    r_ua = -0.65 * r_th
    # Elbow: slight bend relative to upper arm (always flexed a bit)
    # absolute forearm angle = upper + relative flex toward body front
    # relative flex ≈ -40° keeps hand in front of back-swung arm naturally
    l_fa = l_ua - 40.0
    r_fa = r_ua - 40.0

    bones = {
        "body": Affine.translate(0.0, bob),
        "head": Affine.rotate(2.0 * math.sin(2 * math.pi * t / cycle)),
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
    t: float, stride: float = 28.0, bob_amp: float = 5.0, cycle_duration: float = 0.55
) -> dict[str, Affine]:
    return grounded_walk(
        t, step_length=min(stride * 1.5, 40.0), cycle=cycle_duration, bob_amp=bob_amp
    )["bones"]


def path_position(
    t: float, path: Sequence[tuple[float, float]], speed: float = 80.0
) -> tuple[tuple[float, float], float, float]:
    if not path or len(path) < 2:
        p0 = path[0] if path else (0.0, 0.0)
        return p0, 0.0, t * speed
    segs = []
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
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
            return (
                (a[0] + (b[0] - a[0]) * uu, a[1] + (b[1] - a[1]) * uu),
                math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
                dist_raw,
            )
        traveled += length
    return path[-1], 0.0, dist_raw


def sample_viseme_at(cues: Sequence[VisemeCue], t: float) -> tuple[Viseme, float]:
    for cue in cues:
        if cue.timing.contains(t):
            return cue.value, cue.intensity
    return Viseme.X, 0.0


def camera_at(t: float, actions: Sequence, default: CameraState | None = None) -> CameraState:
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
        return CameraState(x=fx + (tx - fx) * u, y=fy + (ty - fy) * u, zoom=z0 + (z1 - z0) * u)
    return default


def parallax_offset(
    camera: CameraState, layer_parallax: float, t: float, scroll_x: float = 0.0, scroll_y: float = 0.0
) -> tuple[float, float]:
    return (-camera.x * layer_parallax + scroll_x * t, -camera.y * layer_parallax + scroll_y * t)


def facing_from_heading(heading_deg: float) -> float:
    return 1.0 if math.cos(math.radians(heading_deg)) >= 0 else -1.0
