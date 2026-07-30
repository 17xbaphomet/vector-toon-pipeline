"""Stream: base → opaque overlays → signs → character (always on top)."""

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
from domain.zones import OVERLAY_FILL, LandscapeRoute, Overlay, generate_route
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

    def _draw_overlay(
        self, canvas: Image.Image, ov: Overlay, body_x: float
    ) -> Image.Image:
        """
        Opaque place scenery locked to world span [start, end].
        Fully covers base inside the span; character is drawn later.
        """
        left = self._world_to_screen_x(ov.start, body_x)
        right = self._world_to_screen_x(ov.end, body_x)
        if self.cfg.facing < 0:
            left, right = right, left

        # Off-screen cull
        if right < -20 or left > self.cfg.width + 20:
            return canvas

        x0 = int(max(-10, left))
        x1 = int(min(self.cfg.width + 10, right))
        if x1 <= x0:
            return canvas

        result = canvas.convert("RGBA")
        panel = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)

        # Opaque fill so base never bleeds through
        fill = OVERLAY_FILL.get(ov.kind, (100, 100, 100, 255))
        draw.rectangle([x0, 0, x1, self.cfg.height], fill=fill)

        # Stack sky + mid + ground, scaled to overlay width
        panel_w = max(1, int(abs(right - left)))
        for path in self.route.overlay_asset_paths(ov.kind):
            if not path.is_file():
                continue
            art = self.renderer._svg_to_pil(path)
            if art.width < 1:
                continue
            scaled = art.resize(
                (panel_w, self.cfg.height), Image.Resampling.LANCZOS
            )
            if scaled.mode != "RGBA":
                scaled = scaled.convert("RGBA")
            # Place so left edge of art sits at world start
            tmp = Image.new("RGBA", result.size, (0, 0, 0, 0))
            tmp.paste(scaled, (int(left), 0), scaled)
            panel = Image.alpha_composite(panel, tmp)

        return Image.alpha_composite(result, panel)

    def _draw_ortsschild(
        self, img: Image.Image, text: str, screen_x: float, ground_y: float
    ) -> Image.Image:
        """Post footed at ground_y — scrolls with world, never floats."""
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

            # 1) Base countryside only
            canvas = self.renderer.render_backgrounds(
                state, (self.cfg.width, self.cfg.height), backgrounds=base
            )

            # 2) Opaque overlays (behind character)
            for ov in self.route.active_overlays(body_x):
                canvas = self._draw_overlay(canvas, ov, body_x)

            # 3) Ground-locked signs (behind character)
            for ov in self.route.active_overlays(body_x, margin=600):
                sx = self._world_to_screen_x(ov.sign_world_x, body_x)
                canvas = self._draw_ortsschild(canvas, ov.sign_text, sx, self._char_sy)

            # 4) Character always on top
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
    print(f"scroll ≈ {spd:.0f} px/s · {n} places · gaps 8k–15k (~1–2 min)")
    for ov in stream.route.overlays[:6]:
        t0 = ov.start / spd
        t1 = ov.end / spd
        print(
            f"  · {ov.kind.value:6s} '{ov.sign_text}'  "
            f"t={t0:.0f}–{t1:.0f}s  world {ov.start:.0f}–{ov.end:.0f}"
        )
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
