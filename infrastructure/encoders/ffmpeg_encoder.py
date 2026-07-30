"""FFmpeg-based video encoder."""

from __future__ import annotations

import subprocess
from pathlib import Path

from domain.interfaces import VideoEncoder


class FFmpegEncoder(VideoEncoder):
    """Mux PNG sequence + audio into MP4 via ffmpeg CLI."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg") -> None:
        self.ffmpeg_bin = ffmpeg_bin

    def encode(
        self,
        frames_dir: Path,
        audio_path: Path,
        output_path: Path,
        fps: int,
    ) -> Path:
        frames_dir = Path(frames_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Expect frame_00000.png, frame_00001.png, ...
        pattern = str(frames_dir / "frame_%05d.png")

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (code {result.returncode}):\n{result.stderr}"
            )
        return output_path
