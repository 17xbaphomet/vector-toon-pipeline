"""Main orchestrator with explicit finite-state machine."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Callable, Mapping, Sequence

from domain.entities import FrameState, SceneSpec, VisemeCue
from domain.interfaces import (
    CharacterAssetRepository,
    FrameRenderer,
    VideoEncoder,
    VisemeExtractor,
)
from domain.value_objects import Affine, Timing, Viseme
from domain.procedural import head_bob, sample_viseme_at
from application.exceptions import PipelineStageError


class PipelineState(Enum):
    PENDING = auto()
    PARSING = auto()
    EXTRACTING_VISEMES = auto()
    GENERATING_CLIPS = auto()
    COMPOSING = auto()
    RENDERING = auto()
    ENCODING = auto()
    DONE = auto()
    FAILED = auto()


ProgressCallback = Callable[[PipelineState, str], None]


class VideoGenerationPipeline:
    """Deterministic offline cartoon video pipeline. All infrastructure injected."""

    def __init__(
        self,
        viseme_extractor: VisemeExtractor,
        asset_repo: CharacterAssetRepository,
        frame_renderer: FrameRenderer,
        video_encoder: VideoEncoder,
        work_dir: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.viseme_extractor = viseme_extractor
        self.asset_repo = asset_repo
        self.frame_renderer = frame_renderer
        self.video_encoder = video_encoder
        self.work_dir = Path(work_dir or "./work")
        self.on_progress = on_progress or (lambda _s, _m: None)
        self.state = PipelineState.PENDING
        self._last_error: Exception | None = None

    def run(self, scene: SceneSpec, output_path: Path) -> Path:
        try:
            self._transition(PipelineState.PARSING)
            self.work_dir.mkdir(parents=True, exist_ok=True)

            self._transition(PipelineState.EXTRACTING_VISEMES)
            visemes = self._extract_visemes(scene)

            self._transition(PipelineState.GENERATING_CLIPS)
            # procedural clips generated on-the-fly during compose for MVP

            self._transition(PipelineState.COMPOSING)
            frames = self._compose_timeline(scene, visemes)

            self._transition(PipelineState.RENDERING)
            frames_dir = self._render_frames(scene, frames)

            self._transition(PipelineState.ENCODING)
            final = self.video_encoder.encode(
                frames_dir=frames_dir,
                audio_path=scene.audio_path,
                output_path=Path(output_path),
                fps=scene.fps,
            )

            self._transition(PipelineState.DONE)
            return final
        except Exception as exc:
            self._last_error = exc
            self.state = PipelineState.FAILED
            self.on_progress(self.state, str(exc))
            raise

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _extract_visemes(self, scene: SceneSpec) -> Sequence[VisemeCue]:
        self.on_progress(self.state, f"Extracting visemes from {scene.audio_path}")
        try:
            return self.viseme_extractor.extract(scene.audio_path)
        except Exception as e:
            raise PipelineStageError("EXTRACTING_VISEMES", str(e), e) from e

    def _compose_timeline(
        self, scene: SceneSpec, visemes: Sequence[VisemeCue]
    ) -> list[FrameState]:
        """Sample at scene.fps: viseme + simple head bob, fixed root for MVP."""
        self.on_progress(self.state, "Composing FrameState sequence")
        frames: list[FrameState] = []
        n = max(1, int(scene.duration * scene.fps))
        dt = 1.0 / scene.fps

        # Use first character for MVP
        char = scene.characters[0] if scene.characters else None
        if char is None:
            return frames

        for i in range(n):
            t = i * dt
            viseme, jaw = sample_viseme_at(visemes, t)
            bob = head_bob(t, amplitude=4.0, freq=2.5)

            bone: dict[str, Affine] = {
                "head": Affine.translate(0.0, bob),
            }

            frames.append(
                FrameState(
                    time=t,
                    character_id=char.id,
                    viseme=viseme,
                    jaw_open=jaw,
                    bone_transforms=bone,
                    root_position=(scene.width / 2, scene.height * 0.65),
                    root_rotation_deg=0.0,
                    scale=char.default_scale,
                )
            )
        return frames

    def _render_frames(
        self, scene: SceneSpec, frames: list[FrameState]
    ) -> Path:
        self.on_progress(self.state, f"Rendering {len(frames)} frames")
        frames_dir = self.work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        rig_cache: dict[str, object] = {}
        for i, state in enumerate(frames):
            if state.character_id not in rig_cache:
                rig_cache[state.character_id] = self.asset_repo.load(state.character_id)
            rig = rig_cache[state.character_id]
            out = frames_dir / f"frame_{i:05d}.png"
            # FrameRenderer returns a path; we copy/rename if needed
            rendered = self.frame_renderer.render(
                state, rig, (scene.width, scene.height)  # type: ignore[arg-type]
            )
            if rendered.resolve() != out.resolve():
                out.write_bytes(rendered.read_bytes())
        return frames_dir

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------

    _ALLOWED: Mapping[PipelineState, set[PipelineState]] = {
        PipelineState.PENDING: {PipelineState.PARSING},
        PipelineState.PARSING: {PipelineState.EXTRACTING_VISEMES, PipelineState.FAILED},
        PipelineState.EXTRACTING_VISEMES: {
            PipelineState.GENERATING_CLIPS,
            PipelineState.FAILED,
        },
        PipelineState.GENERATING_CLIPS: {PipelineState.COMPOSING, PipelineState.FAILED},
        PipelineState.COMPOSING: {PipelineState.RENDERING, PipelineState.FAILED},
        PipelineState.RENDERING: {PipelineState.ENCODING, PipelineState.FAILED},
        PipelineState.ENCODING: {PipelineState.DONE, PipelineState.FAILED},
        PipelineState.DONE: set(),
        PipelineState.FAILED: set(),
    }

    def _transition(self, new: PipelineState) -> None:
        allowed = self._ALLOWED.get(self.state, set())
        if new not in allowed and new != PipelineState.FAILED:
            from application.exceptions import PipelineError

            raise PipelineError(f"Cannot transition {self.state.name} → {new.name}")
        self.state = new
        self.on_progress(new, f"Entered {new.name}")
