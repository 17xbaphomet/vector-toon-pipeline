# vector-toon-pipeline

Offline, deterministic cartoon video generation pipeline that relies on **vector graphics** (SVG / cutouts) and procedural animation.

**No heavy generative image AI** in the render loop.

## Features

- Lip-sync + jaw/head movement driven by **Rhubarb Lip Sync** (Preston Blair 9 shapes)
- **Offline TTS** via Piper (text → WAV → lipsync)
- Procedural walk cycles, path following, vehicle drive
- Multi-layer **parallax backgrounds**
- Layered SVG characters with hierarchical bone/joint transforms
- **Character editor** (Streamlit) for joint placement + movement rules
- Timeline-based scene composition (JSON)
- Frame rendering → FFmpeg MP4

## Architecture (Clean)

```
vector-toon-pipeline/
├── domain/                  # pure entities, value objects, interfaces, procedural
├── application/             # VideoGenerationPipeline (FSM) + exceptions
├── infrastructure/
│   ├── lipsync/             # Rhubarb + energy fallback
│   ├── tts/                 # Piper TTS adapter
│   ├── renderers/           # PillowCutoutRenderer (SVG → PNG + parallax)
│   ├── encoders/            # FFmpegEncoder
│   └── assets/              # FileCharacterAssetRepository
├── tools/
│   └── character_editor.py  # Streamlit joint / rule editor
├── assets/characters/bob/
├── examples/
├── cli.py
└── requirements.txt
```

## Status (GitHub Guardian)

- [x] Domain models (Viseme, Affine, FrameState, CharacterRig, SceneSpec)
- [x] BoneDef, MovementRule, BackgroundLayer, CameraState
- [x] TTSProvider interface
- [x] Rhubarb VisemeExtractor (+ energy fallback)
- [x] Application layer + Pipeline FSM
- [x] domain/procedural.py (head_bob, walk_cycle, path_follower, parallax)
- [x] MVP FrameRenderer (mouth swap + head bob)
- [x] FFmpeg encoder
- [x] Sample character "bob"
- [x] CLI wiring
- [ ] Piper TTS adapter
- [ ] Streamlit character editor (joint placer)
- [ ] Parallax multi-layer backgrounds in renderer
- [ ] Full walk/drive actions in compose stage

## Install

```bash
pip install -r requirements.txt
# Rhubarb binary (optional but recommended):
# https://github.com/DanielSWolf/rhubarb-lip-sync/releases
# FFmpeg on PATH
# Piper voices: python -m piper.download_voices en_US-lessac-medium
```

## Quick start

```bash
# From text (TTS → lipsync → video)
python cli.py examples/demo_scene.json -o out.mp4 --text "Hello, I am Bob the cartoon."

# From existing audio
python cli.py examples/demo_scene.json -o out.mp4 --work-dir ./work

# Character editor
streamlit run tools/character_editor.py
```

## Pipeline stages

1. (Optional) TTS: dialogue → WAV
2. Parse SceneSpec + load CharacterRig (bones + rules)
3. Audio → VisemeCue list (Rhubarb)
4. Generate procedural clips (walk, drive, head bob) from MovementRules + SceneActions
5. Compose timeline → FrameState sequence (incl. camera + parallax)
6. Render frames (background layers → character layers → mouth)
7. Encode MP4 + audio

Everything is data-driven and offline.
