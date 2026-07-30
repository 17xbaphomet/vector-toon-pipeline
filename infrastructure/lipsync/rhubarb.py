"""Rhubarb Lip Sync adapter.

Requires the `rhubarb` binary on PATH (or pass path via constructor).
See https://github.com/DanielSWolf/rhubarb-lip-sync
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Sequence

from domain.entities import VisemeCue
from domain.interfaces import VisemeExtractor
from domain.value_objects import Timing, Viseme

logger = logging.getLogger(__name__)


class RhubarbVisemeExtractor(VisemeExtractor):
    """Extract Preston Blair mouth shapes from audio using the Rhubarb CLI."""

    def __init__(self, rhubarb_bin: str | Path = "rhubarb") -> None:
        self.rhubarb_bin = str(rhubarb_bin)

    def extract(
        self,
        audio_path: Path,
        transcript: str | None = None,
        *,
        work_dir: Path | None = None,
    ) -> Sequence[VisemeCue]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        out_dir = work_dir or audio_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{audio_path.stem}.rhubarb.json"

        cmd: list[str] = [
            self.rhubarb_bin,
            "-f",
            "json",
            "-o",
            str(out_json),
        ]

        dialog_path: Path | None = None
        if transcript:
            dialog_path = out_dir / f"{audio_path.stem}.dialog.txt"
            dialog_path.write_text(transcript, encoding="utf-8")
            cmd.extend(["--dialogFile", str(dialog_path)])

        cmd.append(str(audio_path))

        logger.info("Running Rhubarb: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stderr:
                logger.debug("Rhubarb stderr: %s", result.stderr)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Rhubarb binary not found at '{self.rhubarb_bin}'. "
                "Install from https://github.com/DanielSWolf/rhubarb-lip-sync/releases "
                "and ensure it is on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Rhubarb failed (exit {exc.returncode}): {exc.stderr or exc.stdout}"
            ) from exc

        data = json.loads(out_json.read_text(encoding="utf-8"))
        cues: list[VisemeCue] = []
        for cue in data.get("mouthCues", []):
            value = cue["value"]
            try:
                viseme = Viseme(value)
            except ValueError:
                logger.warning("Unknown viseme '%s', falling back to X", value)
                viseme = Viseme.X
            cues.append(
                VisemeCue(
                    timing=Timing(start=float(cue["start"]), end=float(cue["end"])),
                    value=viseme,
                    intensity=1.0,
                )
            )

        logger.info("Extracted %d viseme cues (duration %.2fs)", len(cues), data.get("metadata", {}).get("duration", 0))
        return cues
