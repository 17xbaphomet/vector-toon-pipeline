# vector-toon-pipeline

Offline, deterministic cartoon video generation pipeline that relies on **vector graphics** (SVG / cutouts) and procedural animation.

**No heavy generative image AI** in the render loop.

## Features (target)

- Lip-sync + jaw/head movement driven by **Rhubarb Lip Sync** (Preston Blair 9 shapes)
- Procedural walk cycles
- Vehicle / path following ("auto fahren")
- Layered SVG characters with hierarchical transforms
- Timeline-based scene composition (JSON/YAML)
- Frame rendering → FFmpeg MP4

## Architecture (Clean)

```
vector-toon-pipeline/
├── domain/              # pure entities, value objects, interfaces
├── application/         # use-cases + pipeline orchestrator
├── infrastructure/      # Rhubarb wrapper, SVG renderer, FFmpeg
├── assets/              # character packs, mouth shapes, backgrounds
├── cli.py
├── requirements.txt
└── README.md
```

## Quick status (GitHub Guardian)

- [x] Domain models (Viseme, Affine, FrameState, CharacterRig, SceneSpec, interfaces)
- [ ] Bootstrap folders + application skeleton
- [ ] Rhubarb VisemeExtractor
- [ ] MVP mouth-swap renderer
- [ ] Procedural walk + path follower
- [ ] Full CLI

## Install

```bash
pip install -r requirements.txt
# Also install Rhubarb Lip Sync binary: https://github.com/DanielSWolf/rhubarb-lip-sync/releases
# and FFmpeg
```

## Pipeline stages

1. Parse SceneSpec + load CharacterRig assets
2. Audio → VisemeCue list (Rhubarb)
3. Generate procedural clips (walk, drive, head bob)
4. Compose timeline → sequence of FrameState
5. Render frames (SVG layers + transforms → PNG)
6. Encode MP4 + original audio

Everything is data-driven and offline.
