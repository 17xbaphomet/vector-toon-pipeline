"""Continuous real-time stream: compute frames on the fly and serve/pipe them."""

from __future__ import annotations

import io
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Iterator, Sequence

from PIL import Image

from domain.entities import CharacterRig, FrameState, SceneSpec
from domain.procedural import grounded_walk, head_bob
from domain.value_objects import Affine, BackgroundLayer, CameraState, Viseme
from infrastructure.renderers.pillow_cutout import PillowCutoutRenderer


@dataclass
class StreamConfig:
    fps: float = 24.0
    width: int = 800
    height: int = 480
    step_length: float = 40.0
    cycle: float = 0.6
    facing: float = 1.0
    scale: float = 1.15
    character_id: str = "bob"
    # Infinite if None
    duration: float | None = None


class ContinuousWalkStream:
    """
    Continuously computes walk + parallax frames in real time.

    Usage:
        stream = ContinuousWalkStream(scene, renderer, rig)
        for jpeg_bytes in stream.frames_jpeg():
            ...  # push to network / ffmpeg
    """

    def __init__(
        self,
        scene: SceneSpec,
        renderer: PillowCutoutRenderer,
        rig: CharacterRig,
        config: StreamConfig | None = None,
    ) -> None:
        self.scene = scene
        self.renderer = renderer
        self.rig = rig
        self.cfg = config or StreamConfig(
            fps=float(scene.fps),
            width=scene.width,
            height=scene.height,
            scale=rig.default_scale if hasattr(rig, "default_scale") else 1.15,
            character_id=rig.id,
        )
        self._t0 = time.perf_counter()
        self._running = False
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()

        # Continuous backgrounds: scroll driven by live time
        walk = grounded_walk(0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle)
        self._scroll_speed = walk["scroll_speed"]
        self._backgrounds = tuple(
            BackgroundLayer(
                path=layer.path,
                z_index=layer.z_index,
                parallax=layer.parallax,
                scroll_x=-self.cfg.facing
                * self._scroll_speed
                * max(layer.parallax, 0.05),
                scroll_y=layer.scroll_y,
                repeat_x=True,
                repeat_y=layer.repeat_y,
            )
            for layer in scene.backgrounds
        )

    def _compose_state(self, t: float) -> FrameState:
        gw = grounded_walk(
            t,
            step_length=self.cfg.step_length,
            cycle=self.cfg.cycle,
            facing=self.cfg.facing,
        )
        bones = dict(gw["bones"])
        bob = head_bob(t, amplitude=2.0, freq=2.5)
        head = bones.get("head", Affine.identity())
        bones["head"] = head.compose(Affine.translate(0.0, bob))

        return FrameState(
            time=t,
            character_id=self.cfg.character_id,
            viseme=Viseme.X,
            jaw_open=0.0,
            bone_transforms=bones,
            root_position=(self.cfg.width * 0.40, self.cfg.height * 0.82),
            root_rotation_deg=0.0 if self.cfg.facing > 0 else 180.0,
            scale=self.cfg.scale * self.cfg.facing,
            camera=CameraState(),
        )

    def frames(self) -> Iterator[Image.Image]:
        """Yield RGB frames paced to real-time fps. Runs until duration or stop()."""
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0

        while self._running:
            t = frame_i * dt
            if self.cfg.duration is not None and t >= self.cfg.duration:
                break

            target = self._t0 + frame_i * dt
            now = time.perf_counter()
            sleep = target - now
            if sleep > 0:
                time.sleep(sleep)

            state = self._compose_state(t)
            img = self.renderer.render_image(
                state,
                self.rig,
                (self.cfg.width, self.cfg.height),
                backgrounds=self._backgrounds,
            )
            yield img
            frame_i += 1

    def frames_jpeg(self, quality: int = 75) -> Iterator[bytes]:
        for img in self.frames():
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            with self._lock:
                self._latest_jpeg = data
            yield data

    def stop(self) -> None:
        self._running = False

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg


def run_mjpeg_server(
    stream: ContinuousWalkStream,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> HTTPServer:
    """
    Start a multipart MJPEG HTTP server.
    Open http://localhost:8765/ in a browser to watch the live stream.
    """

    boundary = b"frame"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (
                    b"<!DOCTYPE html><html><head><title>Vector Toon Stream</title>"
                    b"<style>body{margin:0;background:#111;display:flex;"
                    b"justify-content:center;align-items:center;height:100vh}"
                    b"img{max-width:100%;border:2px solid #333}</style></head>"
                    b"<body><img src='/stream.mjpg' alt='live'></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}"
            )
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                for jpeg in stream.frames_jpeg():
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = HTTPServer((host, port), Handler)
    print(f"MJPEG stream → http://{host}:{port}/  (Ctrl+C to stop)")
    return server


def pipe_to_ffmpeg(
    stream: ContinuousWalkStream,
    output: str = "pipe:1",
    extra_args: Sequence[str] | None = None,
) -> int:
    """
    Encode live frames via ffmpeg stdin (raw JPEG sequence or image2pipe).

    Example:
        python cli.py scene.json --stream --pipe out_live.mp4
    """
    import subprocess

    fps = stream.cfg.fps
    w, h = stream.cfg.width, stream.cfg.height
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(output)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for jpeg in stream.frames_jpeg(quality=80):
            proc.stdin.write(jpeg)
            proc.stdin.flush()
    except BrokenPipeError:
        pass
    finally:
        stream.stop()
        proc.stdin.close()
        return proc.wait()
