"""Continuous stream: base landscape + walk-into overlays, ground-locked signs."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image, ImageDraw, ImageFont

from domain.entities import CharacterRig, FrameState, SceneSpec
from domain.procedural import grounded_walk, head_bob
from domain.value_objects import Affine, CameraState, Viseme
from domain.zones import LandscapeRoute, Overlay, generate_route
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
    duration: float | None = None
    route_seed: int | None = 42


class ContinuousWalkStream:
    """Live walk over continuous countryside with overlay towns/forest."""

    def __init__(
        self,
        scene: SceneSpec,
        renderer: PillowCutoutRenderer,
        rig: CharacterRig,
        config: StreamConfig | None = None,
        route: LandscapeRoute | None = None,
    ) -> None:
        self.scene = scene
        self.renderer = renderer
        self.rig = rig
        self.cfg = config or StreamConfig(
            fps=float(scene.fps),
            width=scene.width,
            height=scene.height,
            scale=getattr(rig, "default_scale", 1.15),
            character_id=rig.id,
        )
        self.route = route or generate_route(
            length=30000.0, seed=self.cfg.route_seed
        )
        self._t0 = time.perf_counter()
        self._running = False
        sample = grounded_walk(
            0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle
        )
        self._scroll_speed = sample["scroll_speed"]
        self._char_screen_x = self.cfg.width * 0.40
        self._char_screen_y = self.cfg.height * 0.82

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
            root_position=(self._char_screen_x, self._char_screen_y),
            root_rotation_deg=0.0 if self.cfg.facing > 0 else 180.0,
            scale=self.cfg.scale * self.cfg.facing,
            camera=CameraState(),
        )

    def _world_to_screen_x(self, world_x: float, body_world_x: float) -> float:
        """Ground-locked: object stays planted on the scrolling ground plane."""
        return self._char_screen_x + (world_x - body_world_x) * self.cfg.facing

    def _draw_overlay(
        self,
        canvas: Image.Image,
        overlay: Overlay,
        body_world_x: float,
    ) -> Image.Image:
        """Blit overlay mid art with left edge locked to overlay.start in world space."""
        layers = self.route.overlay_layers(
            overlay.kind, self._scroll_speed, self.cfg.facing
        )
        result = canvas.convert("RGBA")
        for layer in layers:
            path = Path(layer.path)
            if not path.is_file():
                continue
            img = self.renderer._svg_to_pil(path)
            sx = self._world_to_screen_x(overlay.start, body_world_x)
            sy = 0
            target_w = max(200, int(overlay.width))
            if img.width > 0 and abs(img.width - target_w) > 50:
                scale = target_w / img.width
                nh = max(1, int(img.height * scale))
                img = img.resize((target_w, nh), Image.Resampling.LANCZOS)
            if sx + img.width < -50 or sx > self.cfg.width + 50:
                continue
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            tmp = Image.new("RGBA", result.size, (0, 0, 0, 0))
            tmp.alpha_composite(img, (int(sx), int(sy)))
            result = Image.alpha_composite(result, tmp)
        return result.convert("RGB")

    def _draw_ortsschild(
        self,
        img: Image.Image,
        text: str,
        screen_x: float,
        ground_y: float,
    ) -> Image.Image:
        """Ortsschild planted on the ground — post reaches ground_y, never floats."""
        if screen_x < -80 or screen_x > img.width + 80:
            return img
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = 150, 50
        post_h = 55
        post_top = ground_y - post_h
        plate_bottom = post_top + 6
        plate_top = plate_bottom - h
        cx = int(screen_x)

        draw.rectangle(
            [cx - 4, int(post_top), cx + 4, int(ground_y)],
            fill=(90, 90, 90, 255),
        )
        x0 = cx - w // 2
        y0 = int(plate_top)
        draw.rounded_rectangle(
            [x0, y0, x0 + w, y0 + h],
            radius=4,
            fill=(245, 245, 245, 255),
            outline=(30, 30, 30, 255),
            width=3,
        )
        draw.rounded_rectangle(
            [x0 + 4, y0 + 4, x0 + w - 4, y0 + h - 4],
            radius=2,
            outline=(30, 30, 30, 220),
            width=1,
        )
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, y0 + (h - th) // 2 - 1),
            text,
            fill=(20, 20, 20, 255),
            font=font,
        )
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def frames(self) -> Iterator[Image.Image]:
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0
        facing = self.cfg.facing
        base = self.route.base_layers(self._scroll_speed, facing)

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
            body_world_x = self._scroll_speed * t

            # 1) Continuous base countryside
            img = self.renderer.render_image(
                state,
                self.rig,
                (self.cfg.width, self.cfg.height),
                backgrounds=base,
            )

            # 2) Walk-into overlays — no fade
            for ov in self.route.active_overlays(body_world_x):
                img = self._draw_overlay(img, ov, body_world_x)

            # 3) Ground-locked Ortsschild at overlay entrances
            ground_y = self._char_screen_y
            for ov in self.route.active_overlays(body_world_x, margin=400):
                sx = self._world_to_screen_x(ov.sign_world_x, body_world_x)
                img = self._draw_ortsschild(img, ov.sign_text, sx, ground_y)

            yield img
            frame_i += 1

    def frames_jpeg(self, quality: int = 75) -> Iterator[bytes]:
        for img in self.frames():
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            yield buf.getvalue()

    def stop(self) -> None:
        self._running = False


def run_mjpeg_server(
    stream: ContinuousWalkStream,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> HTTPServer:
    boundary = b"frame"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
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
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={boundary.decode()}",
            )
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            try:
                for jpeg in stream.frames_jpeg():
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = HTTPServer((host, port), Handler)
    n = len(stream.route.overlays)
    print(f"MJPEG stream → http://{host}:{port}/  (Ctrl+C to stop)")
    print(f"Route: continuous countryside + {n} randomized overlays (no fade)")
    for ov in stream.route.overlays[:8]:
        print(
            f"  · {ov.kind.value:6s}  '{ov.sign_text}'  @ world {ov.start:.0f}–{ov.end:.0f}"
        )
    if n > 8:
        print(f"  … +{n - 8} more")
    return server


def pipe_to_ffmpeg(
    stream: ContinuousWalkStream,
    output: str = "pipe:1",
    extra_args: Sequence[str] | None = None,
) -> int:
    import subprocess

    fps = stream.cfg.fps
    cmd = [
        "ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-tune", "zerolatency",
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
