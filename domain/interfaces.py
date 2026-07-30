from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from .entities import CharacterRig, FrameState, VisemeCue
from .value_objects import Viseme


class VisemeExtractor(ABC):
    @abstractmethod
    def extract(self, audio_path: Path, transcript: str | None = None) -> Sequence[VisemeCue]:
        """Return timed viseme cues for the given audio."""


class CharacterAssetRepository(ABC):
    @abstractmethod
    def load(self, character_id: str) -> CharacterRig:
        ...

    @abstractmethod
    def list_ids(self) -> Sequence[str]:
        ...


class FrameRenderer(ABC):
    @abstractmethod
    def render(self, state: FrameState, rig: CharacterRig, canvas_size: tuple[int, int]) -> Path:
        """Render one frame to a PNG path and return it."""


class VideoEncoder(ABC):
    @abstractmethod
    def encode(
        self,
        frames_dir: Path,
        audio_path: Path,
        output_path: Path,
        fps: int,
    ) -> Path:
        """Mux image sequence + audio into final video. Return output path."""
