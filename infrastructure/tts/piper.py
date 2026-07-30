"""Piper TTS adapter – offline, ONNX, CPU-only."""

from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path
from typing import Sequence

from domain.interfaces import TTSProvider

logger = logging.getLogger(__name__)


class PiperTTSProvider(TTSProvider):
    """Uses piper-tts (ONNX). Download: python -m piper.download_voices en_US-lessac-medium"""

    def __init__(self, voices_dir: Path | None = None, piper_bin: str = "piper") -> None:
        self.voices_dir = Path(voices_dir) if voices_dir else Path.home() / ".local/share/piper"
        self.piper_bin = piper_bin

    def list_voices(self) -> Sequence[str]:
        voices: list[str] = []
        for d in [self.voices_dir, Path.home() / ".cache/piper", Path("/usr/share/piper")]:
            if not d.is_dir():
                continue
            for p in d.rglob("*.onnx"):
                voices.append(p.stem)
        return sorted(set(voices)) or ["en_US-lessac-medium"]

    def synthesize(self, text: str, voice_id: str, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._synthesize_python(text, voice_id, out_path)
        except Exception as e:
            logger.debug("Python Piper API failed (%s), trying CLI", e)
            return self._synthesize_cli(text, voice_id, out_path)

    def _find_model(self, voice_id: str) -> tuple[Path, Path | None]:
        candidates = [
            self.voices_dir / f"{voice_id}.onnx",
            self.voices_dir / voice_id / f"{voice_id}.onnx",
            Path.home() / ".cache/piper" / f"{voice_id}.onnx",
        ]
        for onnx in candidates:
            if onnx.is_file():
                cfg = onnx.with_suffix(".onnx.json")
                if not cfg.is_file():
                    cfg = onnx.with_suffix(".json")
                return onnx, cfg if cfg.is_file() else None
        return Path(voice_id), None

    def _synthesize_python(self, text: str, voice_id: str, out_path: Path) -> Path:
        from piper import PiperVoice  # type: ignore

        onnx, cfg = self._find_model(voice_id)
        voice = PiperVoice.load(str(onnx), config_path=str(cfg) if cfg else None)
        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize(text, wav_file)
        logger.info("Piper synthesized %d chars → %s", len(text), out_path)
        return out_path

    def _synthesize_cli(self, text: str, voice_id: str, out_path: Path) -> Path:
        onnx, _ = self._find_model(voice_id)
        cmd = [self.piper_bin, "--model", str(onnx), "--output_file", str(out_path)]
        result = subprocess.run(cmd, input=text, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Piper CLI failed: {result.stderr}\n"
                "Install: pip install piper-tts && python -m piper.download_voices en_US-lessac-medium"
            )
        return out_path
