# vector-toon-pipeline

Offline, deterministic cartoon video generation — **vector SVG + procedural animation**, no heavy image AI.

## Features

- **Piper TTS** → clean WAV → **Rhubarb** lipsync (Preston Blair mouths)
- Procedural **walk cycles** + path following
- Multi-layer **parallax backgrounds**
- **Character editor** (Streamlit) for joints & movement rules
- Timeline JSON scenes → FFmpeg MP4

## Quick start

```bash
pip install -r requirements.txt
python -m piper.download_voices en_US-lessac-medium
# optional: install Rhubarb binary for better lipsync

# From text (TTS → lipsync → video with walk + parallax)
python cli.py examples/demo_scene_walk.json -o out.mp4 --text "Hello, I am Bob!"

# Character editor
streamlit run tools/character_editor.py
```

## Architecture

```
domain/           pure models (BoneDef, BackgroundLayer, SceneSpec, …)
application/      VideoGenerationPipeline FSM
infrastructure/
  tts/            PiperTTSProvider
  lipsync/        Rhubarb + energy fallback
  renderers/      PillowCutoutRenderer (parallax + mouth swap)
  encoders/       FFmpegEncoder
  assets/         FileCharacterAssetRepository
tools/            character_editor.py (Streamlit)
assets/characters/bob/
assets/backgrounds/  sky, hills, ground
```

## Pipeline stages

1. (optional) TTS: dialogue → WAV
2. Viseme extract (Rhubarb)
3. Compose FrameState (walk path, head bob, camera, visemes)
4. Render frames (parallax layers → character → mouth)
5. Encode MP4 + audio

## Status

- [x] Domain (bones, rules, backgrounds, TTS interface)
- [x] Piper TTS adapter
- [x] Rhubarb + fallback
- [x] Pipeline wiring (TTS → lipsync → walk → parallax → encode)
- [x] Streamlit character editor
- [x] CLI `--text` / `--voice`
- [ ] Full hierarchical bone parenting in renderer
- [ ] SVG `data-joint` auto-detection
