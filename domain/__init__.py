"""Pure domain layer for vector-toon-pipeline. No external dependencies beyond stdlib."""

from .value_objects import Viseme, Affine, Timing
from .entities import (
    VisemeCue,
    BoneTransform,
    FrameState,
    CharacterRig,
    SceneAction,
    SceneSpec,
    AnimationClip,
)
from .interfaces import (
    VisemeExtractor,
    CharacterAssetRepository,
    FrameRenderer,
    VideoEncoder,
)

__all__ = [
    "Viseme",
    "Affine",
    "Timing",
    "VisemeCue",
    "BoneTransform",
    "FrameState",
    "CharacterRig",
    "SceneAction",
    "SceneSpec",
    "AnimationClip",
    "VisemeExtractor",
    "CharacterAssetRepository",
    "FrameRenderer",
    "VideoEncoder",
]
