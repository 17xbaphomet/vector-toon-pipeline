"""Main orchestrator with explicit finite-state machine."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Mapping, Sequence

from domain.entities import FrameState, SceneSpec, VisemeCue
from domain.interfaces import (
    CharacterAssetRepository,
    FrameRenderer,
    TTSProvider,
    VideoEncoder,
    VisemeExtractor,
)
from domain.value_objects import Affine, CameraState, Viseme
from domain.procedural import (
    camera_at,
    head_bob,
    path_position,
    sample_viseme_at,
    walk_cycle_pose,
)
from application.exceptions import PipelineStageError


class PipelineState(Enum):
    PENDING = auto()
    PARSING = auto()
    SYNTHESIZING = auto()
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
        tts: TTSProvider | None = None,
        work_dir: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.viseme_extractor = viseme_extractor
        self.asset_repo = asset_repo
        self.frame_renderer = frame_renderer
        self.video_encoder = video_encoder
        self.tts = tts
        self.work_dir = Path(work_dir or "./work")
        self.on_progress = on_progress or (lambda _s, _m: None)
        self.state = PipelineState.PENDING
        self._last_error: Exception | None = None

    def run(self, scene: SceneSpec, output_path: Path) -> Path:
        try:
            self._transition(PipelineState.PARSING)
            self.work_dir.mkdir(parents=True, exist_ok=True)

            # TTS if dialogue present and no audio
            scene = self._maybe_synthesize(scene)

            self._transition(PipelineState.EXTRACTING_VISEMES)
            visemes = self._extract_visemes(scene)

            self._transition(PipelineState.GENERATING_CLIPS)
            # clips generated on-the-fly in compose

            self._transition(PipelineState.COMPOSING)
            frames = self._compose_timeline(scene, visemes)

            self._transition(PipelineState.RENDERING)
            frames_dir = self._render_frames(scene, frames)

            self._transition(PipelineState.ENCODING)
            audio = scene.audio_path
            if audio is None:
                raise PipelineStageError("ENCODING", "No audio_path after TTS/extract")
            final = self.video_encoder.encode(
                frames_dir=frames_dir,
                audio_path=audio,
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

    def _maybe_synthesize(self, scene: SceneSpec) -> SceneSpec:
        if scene.audio_path and Path(scene.audio_path).is_file():
            return scene
        if not scene.dialogue:
            if scene.audio_path is None:
                raise PipelineStageError(
                    "SYNTHESIZING", "Need either audio_path or dialogue text"
                )
            return scene
        if self.tts is None:
            raise PipelineStageError(
                "SYNTHESIZING", "dialogue set but no TTSProvider injected"
            )

        self._transition(PipelineState.SYNTHESIZING)
        out = self.work_dir / "speech.wav"
        self.on_progress(self.state, f"TTS: {scene.dialogue[:60]}…")
        try:
            wav = self.tts.synthesize(scene.dialogue, scene.voice_id, out)
        except Exception as e:
            raise PipelineStageError("SYNTHESIZING", str(e), e) from e
        return replace(scene, audio_path=wav)

    def _extract_visemes(self, scene: SceneSpec) -> Sequence[VisemeCue]:
        assert scene.audio_path is not None
        self.on_progress(self.state, f"Extracting visemes from {scene.audio_path}")
        try:
            transcript = scene.dialogue
            return self.viseme_extractor.extract(scene.audio_path, transcript=transcript)
        except Exception as e:
            raise PipelineStageError("EXTRACTING_VISEMES", str(e), e) from e

    def _compose_timeline(
        self, scene: SceneSpec, visemes: Sequence[VisemeCue]
    ) -> list[FrameState]:
        """Sample at scene.fps: visemes + walk + head bob + camera."""
        self.on_progress(self.state, "Composing FrameState sequence")
        frames: list[FrameState] = []
        n = max(1, int(scene.duration * scene.fps))
        dt = 1.0 / scene.fps

        char = scene.characters[0] if scene.characters else None
        if char is None:
            return frames

        # Load full rig for rules
        try:
            rig = self.asset_repo.load(char.id)
        except Exception:
            rig = char

        walk_params = self._find_action_params(scene, "walk", char.id)
        walk_path = walk_params.get("path") if walk_params else None
        walk_speed = float(walk_params.get("speed", 80)) if walk_params else 80.0

        # Walk rule defaults from character
        stride, cycle, bob_amp = 18.0, 0.6, 6.0
        for rule in rig.rules:
            if rule.type.value == "walk":
                stride = float(rule.params.get("stride", stride))
                cycle = float(rule.params.get("cycle", cycle))
                bob_amp = float(rule.params.get("bob_amp", bob_amp))

        default_pos = (scene.width / 2, scene.height * 0.65)

        for i in range(n):
            t = i * dt
            viseme, jaw = sample_viseme_at(visemes, t)

            # Root position from walk path or center
            if walk_path:
                pts = [tuple(p) for p in walk_path]  # type: ignore[arg-type]
                (rx, ry), heading = path_position(t, pts, speed=walk_speed)
                bones = walk_cycle_pose(t, stride=stride, bob_amp=bob_amp, cycle_duration=cycle)
                root_rot = heading
            else:
                rx, ry = default_pos
                root_rot = 0.0
                bones = {}

            # Head bob (talk or idle)
            bob = head_bob(t, amplitude=4.0, freq=2.5)
            head_tf = bones.get("head", Affine.identity())
            bones["head"] = head_tf.compose(Affine.translate(0.0, bob))

            cam = camera_at(t, scene.actions, CameraState())

            frames.append(
                FrameState(
                    time=t,
                    character_id=char.id,
                    viseme=viseme,
                    jaw_open=jaw,
                    bone_transforms=bones,
                    root_position=(rx, ry),
                    root_rotation_deg=root_rot,
                    scale=char.default_scale,
                    camera=cam,
                )
            )
        return frames

    def _find_action_params(
        self, scene: SceneSpec, action_type: str, character_id: str
    ) -> dict | None:
        for a in scene.actions:
            if a.type == action_type and (
                a.character_id is None or a.character_id == character_id
            ):
                return dict(a.params)
        return None

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
            rendered = self.frame_renderer.render(
                state,
                rig,  # type: ignore[arg-type]
                (scene.width, scene.height),
                backgrounds=scene.backgrounds or None,
            )
            if rendered.resolve() != out.resolve():
                out.write_bytes(rendered.read_bytes())
        return frames_dir

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------

    _ALLOWED: Mapping[PipelineState, set[PipelineState]] = {
        PipelineState.PENDING: {PipelineState.PARSING},
        PipelineState.PARSING: {
            PipelineState.SYNTHESIZING,
            PipelineState.EXTRACTING_VISEMES,
            PipelineState.FAILED,
        },
        PipelineState.SYNTHESIZING: {
            PipelineState.EXTRACTING_VISEMES,
            PipelineState.FAILED,
        },
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
