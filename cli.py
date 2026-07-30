#!/usr/bin/env python3
"""CLI entry point for vector-toon-pipeline (offline + live stream)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
    characters = [
        CharacterRig(
            id=c["id"],
            base_svg=Path(c.get("base_svg", "")),
            layer_paths={},
            mouth_shapes={},
            bone_order=tuple(c.get("bone_order", [])),
            default_scale=float(c.get("default_scale", 1.0)),
        )
        for c in raw.get("characters", [])
    ]
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
    backgrounds = [
        BackgroundLayer(
            path=Path(b["path"]),
            z_index=int(b.get("z_index", 0)),
            parallax=float(b.get("parallax", 1.0)),
            scroll_x=float(b.get("scroll_x", 0)),
            scroll_y=float(b.get("scroll_y", 0)),
            repeat_x=bool(b.get("repeat_x", False)),
            repeat_y=bool(b.get("repeat_y", False)),
        )
        for b in raw.get("backgrounds", [])
    ]
    audio = raw.get("audio_path")
    return SceneSpec(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=int(raw["fps"]),
        duration=float(raw["duration"]),
        audio_path=Path(audio) if audio else None,
        characters=characters,
        actions=actions,
        backgrounds=tuple(backgrounds),
        dialogue=text_override or raw.get("dialogue"),
        voice_id=voice or raw.get("voice_id", "en_US-lessac-medium"),
        background=Path(raw["background"]) if raw.get("background") else None,
    )


def _parse_start_time(s: str | None, tz: str) -> datetime | None:
    if not s:
        return None
    zone = ZoneInfo(tz)
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%H:%M":
                now = datetime.now(zone)
                dt = dt.replace(year=now.year, month=now.month, day=now.day)
            return dt.replace(tzinfo=zone)
        except ValueError:
            continue
    raise SystemExit(f"Invalid --start-time: {s!r} (use YYYY-MM-DDTHH:MM or HH:MM)")


def run_stream(args, scene) -> None:
    from infrastructure.assets.file_repo import FileCharacterAssetRepository
    from infrastructure.renderers.pillow_cutout import PillowCutoutRenderer
    from application.stream import (
        ContinuousWalkStream,
        StreamConfig,
        run_mjpeg_server,
        pipe_to_ffmpeg,
    )

    repo = FileCharacterAssetRepository(ROOT / "assets" / "characters")
    char_id = scene.characters[0].id if scene.characters else "bob"
    try:
        rig = repo.load(char_id)
    except Exception:
        rig = scene.characters[0]

    walk = {}
    for a in scene.actions:
        if a.type == "walk":
            walk = dict(a.params)
            break

    facing = 1.0
    path = walk.get("path")
    if path and len(path) >= 2:
        facing = 1.0 if (path[-1][0] - path[0][0]) >= 0 else -1.0

    cfg = StreamConfig(
        fps=float(scene.fps),
        width=scene.width,
        height=scene.height,
        step_length=float(walk.get("step_length", 40)),
        cycle=float(walk.get("cycle", 0.6)),
        facing=facing,
        scale=float(getattr(rig, "default_scale", 1.15)),
        character_id=char_id,
        duration=None if args.stream_duration <= 0 else args.stream_duration,
        tz=args.tz,
        time_scale=args.time_scale,
        start_time=_parse_start_time(args.start_time, args.tz),
    )

    renderer = PillowCutoutRenderer(cache_dir=args.work_dir / "svg_cache")
    stream = ContinuousWalkStream(scene, renderer, rig, cfg)

    if args.pipe:
        print(f"Piping live frames → ffmpeg → {args.pipe}")
        sys.exit(pipe_to_ffmpeg(stream, output=str(args.pipe)))

    server = run_mjpeg_server(stream, host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping stream…")
        stream.stop()
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vector cartoon pipeline – offline MP4 or live stream"
    )
    parser.add_argument("scene", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--work-dir", type=Path, default=Path("./work"))
    parser.add_argument("--rhubarb", default="rhubarb")
    parser.add_argument("--text", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--stream-duration", type=float, default=0)
    parser.add_argument("--pipe", type=Path, default=None)
    parser.add_argument("--tz", default="Europe/Berlin", help="Timezone for sun/moon")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="1=realtime, 60=1s→1min, 3600=1s→1h",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="YYYY-MM-DDTHH:MM or HH:MM (default: now)",
    )
    args = parser.parse_args()

    if not args.scene.is_file():
        print(f"Scene file not found: {args.scene}")
        sys.exit(1)

    scene = load_scene(args.scene, text_override=args.text, voice=args.voice)
    if args.stream or args.pipe:
        run_stream(args, scene)
        return

    pipeline = build_pipeline(args)
    print(f"Running pipeline → {args.output}")
    print(f"Done: {pipeline.run(scene, args.output)}")


if __name__ == "__main__":
    main()
