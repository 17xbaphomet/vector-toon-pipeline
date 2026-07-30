"""Pure procedural animation helpers (no I/O)."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .entities import VisemeCue
from .value_objects import Affine, CameraState, Viseme


def head_bob(t: float, amplitude: float = 4.0, freq: float = 2.5) -> float:
    return amplitude * math.sin(2 * math.pi * freq * t)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ease_smooth(u: float) -> float:
    """Smoothstep 0..1."""
    u = _clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _two_bone_ik(
    hip: tuple[float, float],
    foot: tuple[float, float],
    thigh_len: float,
    shin_len: float,
    facing: float = 1.0,
) -> tuple[float, float, tuple[float, float]]:
    """
    Analytic 2-bone IK in 2D (profile view).
    Returns (hip_angle_deg, knee_angle_deg, knee_pos).
    Angle 0 = straight down; positive swings toward +x (forward when facing right).
    knee_angle is relative bend (negative = flex).
    """
    dx = foot[0] - hip[0]
    dy = foot[1] - hip[1]
    dist = math.hypot(dx, dy)
    max_reach = thigh_len + shin_len - 0.5
    min_reach = abs(thigh_len - shin_len) + 0.5
    dist = _clamp(dist, min_reach, max_reach)

    # Angle of hip→foot from downward axis
    # down = (0, +1); atan2(x, y) gives angle from down toward +x
    target_angle = math.degrees(math.atan2(dx, dy))

    # Law of cosines at hip
    cos_a = (thigh_len**2 + dist**2 - shin_len**2) / (2 * thigh_len * dist)
    cos_a = _clamp(cos_a, -1.0, 1.0)
    a = math.degrees(math.acos(cos_a))  # angle between thigh and hip-foot line

    # Knee bends "back" relative to facing (behind the leg in profile)
    hip_angle = target_angle - a * facing  # offset so knee goes backward

    # Knee bend angle (interior)
    cos_k = (thigh_len**2 + shin_len**2 - dist**2) / (2 * thigh_len * shin_len)
    cos_k = _clamp(cos_k, -1.0, 1.0)
    knee_interior = math.degrees(math.acos(cos_k))
    knee_angle = knee_interior - 180.0  # 0 = straight, negative = flexed

    rad = math.radians(hip_angle)
    knee_pos = (
        hip[0] + math.sin(rad) * thigh_len,
        hip[1] + math.cos(rad) * thigh_len,
    )
    return hip_angle, knee_angle, knee_pos


def grounded_walk(
    t: float,
    *,
    step_length: float = 55.0,
    cycle: float = 0.6,
    thigh_len: float = 50.0,
    shin_len: float = 48.0,
    hip_height: float = 103.0,
    bob_amp: float = 4.0,
    facing: float = 1.0,
) -> dict:
    """
    Grounded walk cycle – no foot sliding.

    Returns dict with:
      body_world_x: continuous forward progress (world space)
      hip_local: (x, y) hip relative to body root on ground under hip
      bones: Affine rotations for limbs
      feet_local: left/right foot positions relative to body root (for debug)
      scroll_speed: world units/sec body advances (bg should use -facing * this)

    Stance foot is locked in world space; body moves over it.
    Swing foot interpolates between consecutive plant positions with arc.
    """
    # Full cycle = two steps (L plant + R plant)
    step_period = cycle / 2.0
    scroll_speed = step_length / step_period  # constant body velocity

    body_world_x = scroll_speed * t

    # Which step index and phase within current step
    # Even steps: left stance; odd steps: right stance
    total_steps = body_world_x / step_length
    step_i = int(math.floor(total_steps))
    u = total_steps - step_i  # 0..1 within current step

    # Plant positions on ground (world x)
    # Left plants at 0, 2S, 4S, ...
    # Right plants at S, 3S, 5S, ...
    def left_plant_for_step(si: int) -> float:
        # nearest left plant at or behind body progress
        return math.floor(si / 2) * 2 * step_length

    def right_plant_for_step(si: int) -> float:
        return (math.floor((si - 1) / 2) * 2 + 1) * step_length if si >= 1 else -step_length

    if step_i % 2 == 0:
        # Left stance, right swings
        stance_is_left = True
        stance_plant = step_i * step_length  # left plant world x
        # Right swings from previous plant to next
        swing_from = (step_i - 1) * step_length
        swing_to = (step_i + 1) * step_length
    else:
        stance_is_left = False
        stance_plant = step_i * step_length  # right plant
        swing_from = (step_i - 1) * step_length
        swing_to = (step_i + 1) * step_length

    # Foot world positions
    stance_foot_wx = stance_plant
    su = _ease_smooth(u)
    swing_foot_wx = swing_from + (swing_to - swing_from) * su
    # Swing foot lifts in an arc
    swing_lift = 18.0 * math.sin(math.pi * u)

    if stance_is_left:
        left_wx, left_wy = stance_foot_wx, 0.0
        right_wx, right_wy = swing_foot_wx, -swing_lift
    else:
        right_wx, right_wy = stance_foot_wx, 0.0
        left_wx, left_wy = swing_foot_wx, -swing_lift

    # Hip world: above midpoint between feet, with vertical bob
    mid_x = 0.5 * (left_wx + right_wx)
    # Bias hip slightly toward body_world_x for stability
    hip_wx = 0.7 * body_world_x + 0.3 * mid_x
    bob = bob_amp * abs(math.sin(math.pi * total_steps))
    hip_wy = -hip_height + bob  # negative y is up if ground=0; we'll use +down coords

    # Convert to body-local (origin = ground under body_world_x, y+ down)
    def to_local(wx: float, wy: float) -> tuple[float, float]:
        return ((wx - body_world_x) * facing, wy)

    hip_local = to_local(hip_wx, hip_wy)
    # Force hip roughly at x≈0 for centered character
    hip_local = (0.0, -hip_height + bob)

    left_foot = to_local(left_wx, left_wy)
    right_foot = to_local(right_wx, right_wy)

    # When facing left, mirror foot x
    if facing < 0:
        left_foot = (-left_foot[0], left_foot[1])
        right_foot = (-right_foot[0], right_foot[1])

    # IK for both legs (y+ down; hip at hip_local)
    hip_pos = hip_local
    l_hip_a, l_knee_a, _ = _two_bone_ik(
        hip_pos, left_foot, thigh_len, shin_len, facing=1.0
    )
    r_hip_a, r_knee_a, _ = _two_bone_ik(
        hip_pos, right_foot, thigh_len, shin_len, facing=1.0
    )

    # Arms: opposite phase to ipsilateral leg (use hip angles)
    # left arm opposite to left leg
    arm_scale = 0.6
    l_sh = -arm_scale * l_hip_a
    r_sh = -arm_scale * r_hip_a
    l_el = -30.0 - 15.0 * max(0.0, -math.sin(math.pi * total_steps))
    r_el = -30.0 - 15.0 * max(0.0, math.sin(math.pi * total_steps))

    bones: dict[str, Affine] = {
        "body": Affine.translate(0.0, bob),
        "head": Affine.rotate(2.0 * math.sin(2 * math.pi * t / cycle)),
        "left_thigh": Affine.rotate(l_hip_a),
        "left_shin": Affine.rotate(l_knee_a),
        "right_thigh": Affine.rotate(r_hip_a),
        "right_shin": Affine.rotate(r_knee_a),
        "left_upper_arm": Affine.rotate(l_sh),
        "left_forearm": Affine.rotate(l_el),
        "right_upper_arm": Affine.rotate(r_sh),
        "right_forearm": Affine.rotate(r_el),
        "left_leg": Affine.rotate(l_hip_a),
        "right_leg": Affine.rotate(r_hip_a),
        "left_arm": Affine.rotate(l_sh),
        "right_arm": Affine.rotate(r_sh),
    }

    return {
        "body_world_x": body_world_x,
        "scroll_speed": scroll_speed,
        "hip_local": hip_local,
        "bones": bones,
        "feet_local": {"left": left_foot, "right": right_foot},
        "step_length": step_length,
        "cycle": cycle,
    }


def walk_cycle_pose(
    t: float,
    stride: float = 28.0,
    bob_amp: float = 5.0,
    cycle_duration: float = 0.55,
) -> dict[str, Affine]:
    """Legacy wrapper – prefer grounded_walk."""
    result = grounded_walk(
        t,
        step_length=stride * 2.0,
        cycle=cycle_duration,
        bob_amp=bob_amp,
    )
    return result["bones"]


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
    return (
        -camera.x * layer_parallax + scroll_x * t,
        -camera.y * layer_parallax + scroll_y * t,
    )


def facing_from_heading(heading_deg: float) -> float:
    cos_h = math.cos(math.radians(heading_deg))
    return 1.0 if cos_h >= 0 else -1.0
