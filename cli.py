#!/usr/bin/env python3
"""CLI entry point for vector-toon-pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_pipeline(args):
    from infrastructure.lipsync.rhubarb import RhubarbVisemeExtractor
    from infrastructure.lipsync.fallback import EnergyVisemeExtractor
    from infrastructure.assets.file_repo import FileCharacterAssetRepository
    from infrastructure.renderers.pillow_cutout import PillowCutoutRenderer
    from infrastructure.encoders.ffmpeg_encoder import FFmpegEncoder
    from application.pipeline import VideoGenerationPipeline, PipelineState

    # Prefer Rhubarb; fall back to energy-based if binary missing
    try:
        extractor = RhubarbVisemeExtractor(args.rhubarb)
        # quick check
        import shutil
        if not shutil.which(args.rhubarb):
            raise FileNotFoundError
    except Exception:
        print("[warn] Rhubarb not found – using energy fallback visemes")
        extractor = EnergyVisemeExtractor()

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
        work_dir=args.work_dir,
        on_progress=on_progress,
    )


def load_scene(path: Path):
    """Minimal JSON → SceneSpec loader (no pydantic required for MVP)."""
    from domain.entities import CharacterRig, SceneAction, SceneSpec
    from domain.value_objects import Timing, Viseme

    raw = json.loads(path.read_text(encoding="utf-8"))

    characters = []
    for c in raw.get("characters", []):
        # For demo we load via asset repo later; here just stub the id
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

    return SceneSpec(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=int(raw["fps"]),
        duration=float(raw["duration"]),
        audio_path=Path(raw["audio_path"]),
        characters=characters,
        actions=actions,
        background=Path(raw["background"]) if raw.get("background") else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline vector cartoon video pipeline")
    parser.add_argument("scene", type=Path, help="Path to SceneSpec JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--work-dir", type=Path, default=Path("./work"))
    parser.add_argument("--rhubarb", default="rhubarb", help="Rhubarb binary path")
    args = parser.parse_args()

    if not args.scene.is_file():
        print(f"Scene file not found: {args.scene}")
        sys.exit(1)

    scene = load_scene(args.scene)
    pipeline = build_pipeline(args)

    print(f"Running pipeline → {args.output}")
    result = pipeline.run(scene, args.output)
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
