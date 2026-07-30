from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .value_objects import (
    Affine,
    BackgroundLayer,
    BoneDef,
    CameraState,
    MovementRule,
    Timing,
    Viseme,
)


@dataclass(frozen=True, slots=True)
class VisemeCue:
    timing: Timing
    value: Viseme
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
    camera: CameraState = field(default_factory=CameraState)


@dataclass(frozen=True, slots=True)
class AnimationClip:
    """Reusable timed sequence of bone transforms (walk, idle, etc.)."""

    name: str
    duration: float
    samples: Sequence[tuple[float, Mapping[str, Affine]]]


@dataclass(frozen=True, slots=True)
class CharacterRig:
    id: str
    base_svg: Path
    layer_paths: Mapping[str, Path]  # e.g. "head", "body", "left_leg"
    mouth_shapes: Mapping[Viseme, Path]
    bone_order: Sequence[str] = ()
    bones: Sequence[BoneDef] = ()
    rules: Sequence[MovementRule] = ()
    default_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class SceneAction:
    type: str  # "talk" | "walk" | "drive" | "idle" | "camera" | "pan"
    timing: Timing
    character_id: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)
    # walk params: path=[(x,y),...], speed=80
    # drive params: path=..., speed=...
    # camera params: from=(x,y), to=(x,y), zoom=1.0


@dataclass(frozen=True, slots=True)
class SceneSpec:
    width: int
    height: int
    fps: int
    duration: float
    audio_path: Path | None  # None if dialogue will be synthesized
    characters: Sequence[CharacterRig]
    actions: Sequence[SceneAction]
    backgrounds: Sequence[BackgroundLayer] = ()
    dialogue: str | None = None  # if set, TTS runs before lipsync
    voice_id: str = "en_US-lessac-medium"
    background: Path | None = None  # legacy single bg (prefer backgrounds)
