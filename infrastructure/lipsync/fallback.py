"""Simple pure-Python fallback viseme extractor (energy-based, low quality).

Useful when Rhubarb binary is unavailable. Not production quality.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from domain.entities import VisemeCue
from domain.interfaces import VisemeExtractor
from domain.value_objects import Timing, Viseme


class EnergyVisemeExtractor(VisemeExtractor):
    """Very crude amplitude-based mouth open/close. For testing only."""

    def extract(self, audio_path: Path, transcript: str | None = None) -> Sequence[VisemeCue]:
        data, sr = sf.read(str(audio_path))
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Simple frame energy
        frame_len = int(0.05 * sr)  # 50 ms
        hop = frame_len // 2
        energies = []
        times = []
        for i in range(0, len(data) - frame_len, hop):
            frame = data[i : i + frame_len]
            energies.append(float(np.sqrt(np.mean(frame**2))))
            times.append(i / sr)

        if not energies:
            return [VisemeCue(Timing(0.0, 0.1), Viseme.X)]

        energies = np.array(energies)
        threshold = np.percentile(energies, 40)

        cues: list[VisemeCue] = []
        current = Viseme.X
        start = 0.0
        for t, e in zip(times, energies):
            new_v = Viseme.C if e > threshold else Viseme.X  # open-ish vs closed
            if new_v != current:
                if t > start:
                    cues.append(VisemeCue(Timing(start, t), current))
                current = new_v
                start = t
        # final
        duration = len(data) / sr
        cues.append(VisemeCue(Timing(start, duration), current))
        return cues
