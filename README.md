# vector-toon-pipeline

Offline, deterministic cartoon video generation pipeline that relies on **vector graphics** (SVG / cutouts) and procedural animation.

**No heavy generative image AI** in the render loop.

## Features (target)

- Lip-sync + jaw/head movement driven by **Rhubarb Lip Sync** (Preston Blair 9 shapes)
- Procedural walk cycles
- Vehicle / path following ("auto fahren")
- Layered SVG characters with hierarchical transforms
- Timeline-based scene composition (JSON)
- Frame rendering → FFmpeg MP4

## Architecture (Clean)

```
vector-toon-pipeline/
├── domain/                  # pure entities, value objects, interfaces, procedural
├── application/             # VideoGenerationPipeline (FSM) + exceptions
├── infrastructure/
│   ├── lipsync/             # Rhubarb + energy fallback
│   ├── renderers/           # PillowCutoutRenderer (SVG → PNG)
│   ├── encoders/            # FFmpegEncoder
│   └── assets/              # FileCharacterAssetRepository
├── assets/characters/bob/   # sample geometric character + mouth shapes
├── examples/
├── cli.py
├── requirements.txt
└── README.md
```

## Status (GitHub Guardian)

- [x] Domain models (Viseme, Affine, FrameState, CharacterRig, SceneSpec, interfaces)
- [x] Rhubarb VisemeExtractor (+ energy fallback)
- [x] Application layer + Pipeline FSM
- [x] domain/procedural.py (head_bob, walk_cycle, path_follower, sample_viseme)
- [x] MVP FrameRenderer (Pillow + cairosvg mouth swap + head bob)
- [x] FFmpeg encoder
- [x] Sample character "bob" (geometric SVG + 9 mouth shapes)
- [x] CLI wiring
- [ ] Full walk/drive actions in compose stage
- [ ] Better layered SVG hierarchy / bone parenting
- [ ] Headless browser renderer option (playwright)

## Install

```bash
pip install -r requirements.txt
# Rhubarb Lip Sync binary (optional but recommended):
# https://github.com/DanielSWolf/rhubarb-lip-sync/releases
# FFmpeg must be on PATH
```

## Quick start (once you have a short .wav)

```bash
# put a sample.wav next to the demo scene or edit the path
python cli.py examples/demo_scene.json -o out.mp4 --work-dir ./work
```

## Pipeline stages

1. Parse SceneSpec + load CharacterRig assets
2. Audio → VisemeCue list (Rhubarb or energy fallback)
3. Generate procedural clips (walk, drive, head bob)
4. Compose timeline → sequence of FrameState
5. Render frames (SVG layers + transforms → PNG)
6. Encode MP4 + original audio

Everything is data-driven and offline.
