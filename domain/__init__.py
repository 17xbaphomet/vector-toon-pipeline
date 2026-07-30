"""Pure domain layer for vector-toon-pipeline. No external dependencies beyond stdlib."""

from .value_objects import (
    Viseme,
    Affine,
    Timing,
    BoneDef,
    MovementRule,
    MovementRuleType,
    BackgroundLayer,
    CameraState,
)
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
    TTSProvider,
    CharacterAssetRepository,
    FrameRenderer,
    VideoEncoder,
)
from .procedural import head_bob, walk_cycle_pose, path_position, sample_viseme_at

__all__ = [
    "Viseme",
    "Affine",
    "Timing",
    "BoneDef",
    "MovementRule",
    "MovementRuleType",
    "BackgroundLayer",
    "CameraState",
    "VisemeCue",
    "BoneTransform",
    "FrameState",
    "CharacterRig",
    "SceneAction",
    "SceneSpec",
    "AnimationClip",
    "VisemeExtractor",
    "TTSProvider",
    "CharacterAssetRepository",
    "FrameRenderer",
    "VideoEncoder",
    "head_bob",
    "walk_cycle_pose",
    "path_position",
    "sample_viseme_at",
]
