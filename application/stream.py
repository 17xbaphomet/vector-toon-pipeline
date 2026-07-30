"""Stream: base landscape always on; overlays = objects only."""

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
            length=80000.0, seed=self.cfg.route_seed
        )
        self._t0 = time.perf_counter()
        self._running = False
        sample = grounded_walk(
            0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle
        )
        self._scroll_speed = sample["scroll_speed"]
        self._char_sx = self.cfg.width * 0.40
        self._char_sy = self.cfg.height * 0.82

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
            root_position=(self._char_sx, self._char_sy),
            root_rotation_deg=0.0 if self.cfg.facing > 0 else 180.0,
            scale=self.cfg.scale * self.cfg.facing,
            camera=CameraState(),
        )

    def _world_to_screen_x(self, world_x: float, body_x: float) -> float:
        return self._char_sx + (world_x - body_x) * self.cfg.facing

    def _draw_objects(
        self, canvas: Image.Image, ov: Overlay, body_x: float
    ) -> Image.Image:
        """
        Paste transparent object strip (houses/trees/buildings only).
        No sky/ground fill — base landscape stays visible between objects.
        """
        path = self.route.overlay_object_path(ov.kind)
        if not path.is_file():
            return canvas

        left = self._world_to_screen_x(ov.start, body_x)
        right = self._world_to_screen_x(ov.end, body_x)
        if self.cfg.facing < 0:
            left, right = right, left

        if right < -40 or left > self.cfg.width + 40:
            return canvas

        art = self.renderer._svg_to_pil(path)
        panel_w = max(1, int(abs(right - left)))
        # Keep aspect: scale width to overlay span, height stays canvas height
        scaled = art.resize((panel_w, self.cfg.height), Image.Resampling.LANCZOS)
        if scaled.mode != "RGBA":
            scaled = scaled.convert("RGBA")

        result = canvas.convert("RGBA")
        tmp = Image.new("RGBA", result.size, (0, 0, 0, 0))
        tmp.paste(scaled, (int(left), 0), scaled)
        return Image.alpha_composite(result, tmp)

    def _draw_ortsschild(
        self, img: Image.Image, text: str, screen_x: float, ground_y: float
    ) -> Image.Image:
        if screen_x < -100 or screen_x > img.width + 100:
            return img
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        w, h = 150, 48
        post_h = 50
        cx = int(screen_x)
        post_top = ground_y - post_h
        plate_top = post_top - h + 8

        draw.rectangle(
            [cx - 4, int(post_top), cx + 4, int(ground_y)], fill=(80, 80, 80, 255)
        )
        x0 = cx - w // 2
        y0 = int(plate_top)
        draw.rounded_rectangle(
            [x0, y0, x0 + w, y0 + h],
            radius=4,
            fill=(245, 245, 245, 255),
            outline=(25, 25, 25, 255),
            width=3,
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
        return Image.alpha_composite(img.convert("RGBA"), layer)

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
            body_x = self._scroll_speed * t

            # 1) Continuous base: sky + landscape + road (always)
            canvas = self.renderer.render_backgrounds(
                state, (self.cfg.width, self.cfg.height), backgrounds=base
            )

            # 2) Objects only (houses / trees / buildings) — sky shows through
            for ov in self.route.active_overlays(body_x):
                canvas = self._draw_objects(canvas, ov, body_x)

            # 3) Ground-locked Ortsschild
            for ov in self.route.active_overlays(body_x, margin=600):
                sx = self._world_to_screen_x(ov.sign_world_x, body_x)
                canvas = self._draw_ortsschild(canvas, ov.sign_text, sx, self._char_sy)

            # 4) Character on top
            char = self.renderer.render_character(
                state, self.rig, (self.cfg.width, self.cfg.height)
            )
            frame = Image.alpha_composite(canvas.convert("RGBA"), char)

            yield frame.convert("RGB")
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
    spd = stream._scroll_speed
    print(f"MJPEG stream → http://{host}:{port}/  (Ctrl+C to stop)")
    print(f"Objects-only overlays · base sky/landscape always visible · {n} places")
    for ov in stream.route.overlays[:6]:
        t0, t1 = ov.start / spd, ov.end / spd
        print(f"  · {ov.kind.value:6s} '{ov.sign_text}'  t={t0:.0f}–{t1:.0f}s")
    if n > 6:
        print(f"  … +{n - 6} more")
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
