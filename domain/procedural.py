"""Pure procedural animation helpers (no I/O, no external deps beyond math)."""

from __future__ import annotations

import math
from typing import Sequence

from .entities import VisemeCue
from .value_objects import Affine, Viseme


def head_bob(t: float, amplitude: float = 4.0, freq: float = 2.5) -> float:
    """Simple vertical head bob in pixels."""
    return amplitude * math.sin(2 * math.pi * freq * t)


def walk_cycle_pose(
    t: float,
    stride: float = 18.0,
    bob_amp: float = 6.0,
    cycle_duration: float = 0.6,
) -> dict[str, Affine]:
    """
    Return bone transforms for a basic bipedal walk cycle.
    Bones expected: left_leg, right_leg, left_arm, right_arm, body.
    """
    phase = (t % cycle_duration) / cycle_duration  # 0..1
    angle = phase * 2 * math.pi

    leg_swing = stride * math.sin(angle)
    arm_swing = stride * 0.7 * math.sin(angle + math.pi)  # opposite
    body_bob = bob_amp * abs(math.sin(angle))

    return {
        "body": Affine.translate(0.0, -body_bob),
        "left_leg": Affine.rotate(leg_swing * 0.8, cx=0, cy=0),
        "right_leg": Affine.rotate(-leg_swing * 0.8, cx=0, cy=0),
        "left_arm": Affine.rotate(arm_swing * 0.6, cx=0, cy=0),
        "right_arm": Affine.rotate(-arm_swing * 0.6, cx=0, cy=0),
    }


def path_position(
    t: float,
    path: Sequence[tuple[float, float]],
    speed: float = 80.0,
) -> tuple[tuple[float, float], float]:
    """
    Follow a polyline path at constant speed.
    Returns ((x, y), heading_degrees).
    """
    if not path or len(path) < 2:
        return (path[0] if path else (0.0, 0.0)), 0.0

    # Pre-compute segment lengths
    segs: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        segs.append((a, b, length))
        total += length

    if total <= 0:
        return path[0], 0.0

    dist = (t * speed) % total
    traveled = 0.0
    for a, b, length in segs:
        if traveled + length >= dist:
            u = (dist - traveled) / length if length > 0 else 0.0
            x = a[0] + (b[0] - a[0]) * u
            y = a[1] + (b[1] - a[1]) * u
            heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            return (x, y), heading
        traveled += length

    return path[-1], 0.0


def sample_viseme_at(
    cues: Sequence[VisemeCue], t: float
) -> tuple[Viseme, float]:
    """Return (viseme, jaw_open) active at time t."""
    for cue in cues:
        if cue.timing.contains(t):
            return cue.value, cue.intensity
    return Viseme.X, 0.0
