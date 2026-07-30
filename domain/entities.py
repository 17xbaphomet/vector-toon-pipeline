from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .value_objects import Affine, Timing, Viseme


@dataclass(frozen=True, slots=True)
class VisemeCue:
    timing: Timing
    value: Viseme
    # optional intensity for jaw openness 0..1 (Rhubarb does not provide; can be derived)
    intensity: float = 1.0


@dataclass(frozen=True, slots=True)
class BoneTransform:
    bone_id: str
    transform: Affine


@dataclass(frozen=True, slots=True)
class FrameState:
    """Complete pose of one character at one point in time."""

    time: float
    character_id: str
    viseme: Viseme
    jaw_open: float  # 0.0 – 1.0
    bone_transforms: Mapping[str, Affine]
    root_position: tuple[float, float]
    root_rotation_deg: float
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class AnimationClip:
    """Reusable timed sequence of bone transforms (walk, idle, etc.)."""

    name: str
    duration: float
    # key times -> bone_id -> Affine  (or denser samples)
    samples: Sequence[tuple[float, Mapping[str, Affine]]]


@dataclass(frozen=True, slots=True)
class CharacterRig:
    id: str
    base_svg: Path
    # layer / group ids inside SVG or separate files
    layer_paths: Mapping[str, Path]  # e.g. "head", "body", "left_leg"
    mouth_shapes: Mapping[Viseme, Path]
    # optional bone hierarchy for later skeletal
    bone_order: Sequence[str] = ()
    default_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class SceneAction:
    type: str  # "talk" | "walk" | "drive" | "idle" | "camera"
    timing: Timing
    character_id: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneSpec:
    width: int
    height: int
    fps: int
    duration: float
    audio_path: Path
    characters: Sequence[CharacterRig]
    actions: Sequence[SceneAction]
    background: Path | None = None
