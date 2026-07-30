#!/usr/bin/env python3
"""CLI entry point for vector-toon-pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_pipeline(args):
    from infrastructure.lipsync.rhubarb import RhubarbVisemeExtractor
    from infrastructure.lipsync.fallback import EnergyVisemeExtractor
    from infrastructure.assets.file_repo import FileCharacterAssetRepository
    from infrastructure.renderers.pillow_cutout import PillowCutoutRenderer
    from infrastructure.encoders.ffmpeg_encoder import FFmpegEncoder
    from infrastructure.tts.piper import PiperTTSProvider
    from application.pipeline import VideoGenerationPipeline, PipelineState

    try:
        import shutil

        extractor = RhubarbVisemeExtractor(args.rhubarb)
        if not shutil.which(args.rhubarb):
            raise FileNotFoundError
    except Exception:
        print("[warn] Rhubarb not found – using energy fallback visemes")
        extractor = EnergyVisemeExtractor()

    tts = PiperTTSProvider()
    repo = FileCharacterAssetRepository(ROOT / "assets" / "characters")
    renderer = PillowCutoutRenderer(cache_dir=args.work_dir / "svg_cache")
    encoder = FFmpegEncoder()

    def on_progress(state: PipelineState, msg: str) -> None:
        print(f"[{state.name}] {msg}")

    return VideoGenerationPipeline(
        viseme_extractor=extractor,
        asset_repo=repo,
        frame_renderer=renderer,
        video_encoder=encoder,
        tts=tts,
        work_dir=args.work_dir,
        on_progress=on_progress,
    )


def load_scene(path: Path, text_override: str | None = None, voice: str | None = None):
    from domain.entities import CharacterRig, SceneAction, SceneSpec
    from domain.value_objects import BackgroundLayer, Timing

    raw = json.loads(path.read_text(encoding="utf-8"))

    characters = []
    for c in raw.get("characters", []):
        characters.append(
            CharacterRig(
                id=c["id"],
                base_svg=Path(c.get("base_svg", "")),
                layer_paths={},
                mouth_shapes={},
                bone_order=tuple(c.get("bone_order", [])),
                default_scale=float(c.get("default_scale", 1.0)),
            )
        )

    actions = []
    for a in raw.get("actions", []):
        t = a["timing"]
        actions.append(
            SceneAction(
                type=a["type"],
                timing=Timing(start=float(t["start"]), end=float(t["end"])),
                character_id=a.get("character_id"),
                params=a.get("params", {}),
            )
        )

    backgrounds = []
    for b in raw.get("backgrounds", []):
        backgrounds.append(
            BackgroundLayer(
                path=Path(b["path"]),
                z_index=int(b.get("z_index", 0)),
                parallax=float(b.get("parallax", 1.0)),
                scroll_x=float(b.get("scroll_x", 0)),
                scroll_y=float(b.get("scroll_y", 0)),
                repeat_x=bool(b.get("repeat_x", False)),
                repeat_y=bool(b.get("repeat_y", False)),
            )
        )

    audio = raw.get("audio_path")
    audio_path = Path(audio) if audio else None

    dialogue = text_override or raw.get("dialogue")
    voice_id = voice or raw.get("voice_id", "en_US-lessac-medium")

    return SceneSpec(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=int(raw["fps"]),
        duration=float(raw["duration"]),
        audio_path=audio_path,
        characters=characters,
        actions=actions,
        backgrounds=tuple(backgrounds),
        dialogue=dialogue,
        voice_id=voice_id,
        background=Path(raw["background"]) if raw.get("background") else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline vector cartoon video pipeline")
    parser.add_argument("scene", type=Path, help="Path to SceneSpec JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--work-dir", type=Path, default=Path("./work"))
    parser.add_argument("--rhubarb", default="rhubarb", help="Rhubarb binary path")
    parser.add_argument("--text", default=None, help="Override dialogue (triggers TTS)")
    parser.add_argument("--voice", default=None, help="Piper voice id")
    args = parser.parse_args()

    if not args.scene.is_file():
        print(f"Scene file not found: {args.scene}")
        sys.exit(1)

    scene = load_scene(args.scene, text_override=args.text, voice=args.voice)
    pipeline = build_pipeline(args)

    print(f"Running pipeline → {args.output}")
    result = pipeline.run(scene, args.output)
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
