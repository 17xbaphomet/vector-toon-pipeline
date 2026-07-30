# vector-toon-pipeline

Offline, deterministic cartoon video generation — **vector SVG + procedural animation**, no heavy image AI.

## Features

- **Piper TTS** (multi-language, incl. **German**) → WAV → **Rhubarb** lipsync
- Procedural walk cycles + path following
- Multi-layer parallax backgrounds
- Character editor (Streamlit) for joints & movement rules
- Timeline JSON scenes → FFmpeg MP4

## Quick start

```bash
pip install -r requirements.txt

# English voice
python -m piper.download_voices en_US-lessac-medium

# German voice (recommended: Thorsten medium)
python -m piper.download_voices de_DE-thorsten-medium

# German demo (walk + talk + parallax)
python cli.py examples/demo_scene_de.json -o out_de.mp4 --work-dir ./work

# Or override text/voice on any scene
python cli.py examples/demo_scene_walk.json -o out.mp4 \
  --text "Guten Tag, ich bin ein Cartoon!" \
  --voice de_DE-thorsten-medium
```

### German Piper voices (`de_DE`)

| Voice | Quality | Notes |
|-------|---------|-------|
| `de_DE-thorsten-medium` | ★★★★ | Best default, male |
| `de_DE-thorsten-high` | ★★★★★ | Highest quality |
| `de_DE-thorsten_emotional-medium` | ★★★★ | Expressive |
| `de_DE-kerstin-low` | ★★★ | Female |
| `de_DE-ramona-low` | ★★★ | Female |
| `de_DE-karlsson-low` | ★★★ | Male |
| `de_DE-pavoque-low` | ★★★ | Clear |
| `de_DE-eva_k-x_low` | ★★ | Tiny model |
| `de_DE-mls-medium` | ★★★ | Multi-speaker |

Lipsync (Rhubarb) is language-agnostic — German audio works the same as English.

## Character editor

```bash
streamlit run tools/character_editor.py
```

## Architecture

```
domain/           pure models
application/      VideoGenerationPipeline FSM
infrastructure/
  tts/            PiperTTSProvider (en, de, …)
  lipsync/        Rhubarb + energy fallback
  renderers/      PillowCutoutRenderer (parallax + mouth)
  encoders/       FFmpegEncoder
  assets/         FileCharacterAssetRepository
tools/            character_editor.py
assets/characters/bob/
assets/backgrounds/
```

## Pipeline stages

1. (optional) TTS: dialogue → WAV (any Piper language)
2. Viseme extract (Rhubarb)
3. Compose FrameState (walk, head bob, camera)
4. Render (parallax layers → character → mouth)
5. Encode MP4 + audio
