"""Stream: sky behind all → landscape → objects → character → day/night grade."""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator, Sequence
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from domain.entities import CharacterRig, FrameState, SceneSpec
from domain.procedural import grounded_walk, head_bob
from domain.sky import (
    alt_az_to_screen,
    celestial_at,
    scene_grade,
    sky_colors,
)
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
    tz: str = "Europe/Berlin"
    time_scale: float = 1.0
    start_time: datetime | None = None


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
        self.route = route or generate_route(length=80000.0, seed=self.cfg.route_seed)
        self._t0 = time.perf_counter()
        self._running = False
        sample = grounded_walk(0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle)
        self._scroll_speed = sample["scroll_speed"]
        self._char_sx = self.cfg.width * 0.40
        self._char_sy = self.cfg.height * 0.82
        tz = ZoneInfo(self.cfg.tz)
        self._clock0 = self.cfg.start_time or datetime.now(tz)
        if self._clock0.tzinfo is None:
            self._clock0 = self._clock0.replace(tzinfo=tz)

    def _scene_datetime(self, stream_t: float) -> datetime:
        return self._clock0 + timedelta(seconds=stream_t * self.cfg.time_scale)

    def _compose_state(self, t: float) -> FrameState:
        gw = grounded_walk(
            t, step_length=self.cfg.step_length, cycle=self.cfg.cycle, facing=self.cfg.facing
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

    def _make_sky_canvas(self, cel) -> Image.Image:
        w, h = self.cfg.width, self.cfg.height
        top, bot = sky_colors(cel)
        sky = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(sky)
        for y in range(h):
            t = y / max(h - 1, 1)
            if t < 0.55:
                u = t / 0.55
                r = int(top[0] + (bot[0] - top[0]) * u)
                g = int(top[1] + (bot[1] - top[1]) * u)
                b = int(top[2] + (bot[2] - top[2]) * u)
            else:
                u = (t - 0.55) / 0.45
                r = int(bot[0] * (1 - 0.15 * u))
                g = int(bot[1] * (1 - 0.15 * u))
                b = int(bot[2] * (1 - 0.15 * u))
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        return sky.convert("RGBA")

    def _draw_sun(self, canvas: Image.Image, cel) -> Image.Image:
        pos = alt_az_to_screen(cel.sun_alt_deg, cel.sun_az_deg, canvas.width, canvas.height)
        if pos is None:
            return canvas
        x, y = pos
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        r = 28
        for i, alpha in ((18, 40), (12, 80), (0, 255)):
            rr = r + i
            color = (255, 220, 80, alpha) if i else (255, 230, 100, 255)
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=color)
        return Image.alpha_composite(canvas, layer)

    def _draw_moon(self, canvas: Image.Image, cel) -> Image.Image:
        """Lit side opaque; dark side highly transparent so sky shows through by day."""
        pos = alt_az_to_screen(cel.moon_alt_deg, cel.moon_az_deg, canvas.width, canvas.height)
        if pos is None:
            return canvas
        x, y = pos
        r = 22
        moon_lit = (230, 230, 210, 240)
        # Dark side: almost fully transparent — sky bleeds through (no black disc)
        moon_dark = (230, 230, 210, 18)
        phase = cel.moon_phase
        pa = phase * 2.0 * math.pi
        moon_img = Image.new("RGBA", (r * 2 + 2, r * 2 + 2), (0, 0, 0, 0))
        cx, cy = r + 1, r + 1
        for py in range(r * 2 + 2):
            for px in range(r * 2 + 2):
                dx, dy = px - cx, py - cy
                if dx * dx + dy * dy > r * r:
                    continue
                half = math.sqrt(max(0.0, r * r - dy * dy))
                term = math.cos(pa) * half
                lit = dx >= term if phase <= 0.5 else dx <= term
                moon_img.putpixel((px, py), moon_lit if lit else moon_dark)
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(moon_img, (int(x - r - 1), int(y - r - 1)), moon_img)
        return Image.alpha_composite(canvas, layer)

    def _draw_stars(self, canvas: Image.Image, cel) -> Image.Image:
        if cel.sun_alt_deg > -4:
            return canvas
        strength = min(1.0, (-cel.sun_alt_deg - 4) / 10.0)
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        w, h = canvas.size
        horizon = int(h * 0.55)
        state = 42
        for _ in range(60):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            sx = state % w
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            sy = state % horizon
            draw.ellipse([sx, sy, sx + 1, sy + 1], fill=(255, 255, 240, int(180 * strength)))
        return Image.alpha_composite(canvas, layer)

    def _apply_grade(self, img: Image.Image, cel) -> Image.Image:
        grade = scene_grade(cel)
        rgb = img.convert("RGB")
        if abs(grade.brightness - 1.0) > 0.01:
            rgb = ImageEnhance.Brightness(rgb).enhance(grade.brightness)
        if abs(grade.saturation - 1.0) > 0.01:
            rgb = ImageEnhance.Color(rgb).enhance(grade.saturation)
        if any(abs(v) > 0.005 for v in (grade.tint_r, grade.tint_g, grade.tint_b)):
            import numpy as np

            arr = np.asarray(rgb, dtype=np.float32)
            arr[..., 0] = np.clip(arr[..., 0] + grade.tint_r * 255, 0, 255)
            arr[..., 1] = np.clip(arr[..., 1] + grade.tint_g * 255, 0, 255)
            arr[..., 2] = np.clip(arr[..., 2] + grade.tint_b * 255, 0, 255)
            rgb = Image.fromarray(arr.astype(np.uint8), "RGB")
        return rgb

    def _draw_objects(self, canvas: Image.Image, ov: Overlay, body_x: float) -> Image.Image:
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
        cx = int(screen_x)
        post_top = ground_y - 50
        plate_top = post_top - h + 8
        draw.rectangle([cx - 4, int(post_top), cx + 4, int(ground_y)], fill=(80, 80, 80, 255))
        x0, y0 = cx - w // 2, int(plate_top)
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
        draw.text((cx - tw // 2, y0 + (h - th) // 2 - 1), text, fill=(20, 20, 20, 255), font=font)
        return Image.alpha_composite(img.convert("RGBA"), layer)

    def frames(self) -> Iterator[Image.Image]:
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0
        base = self.route.base_layers(self._scroll_speed, self.cfg.facing)

        while self._running:
            t = frame_i * dt
            if self.cfg.duration is not None and t >= self.cfg.duration:
                break
            target = self._t0 + frame_i * dt
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

            state = self._compose_state(t)
            body_x = self._scroll_speed * t
            cel = celestial_at(self._scene_datetime(t), tz=self.cfg.tz)

            canvas = self._make_sky_canvas(cel)
            canvas = self._draw_stars(canvas, cel)
            canvas = self._draw_moon(canvas, cel)
            canvas = self._draw_sun(canvas, cel)
            canvas = self._composite_landscape(canvas, state, base)

            for ov in self.route.active_overlays(body_x):
                canvas = self._draw_objects(canvas, ov, body_x)
            for ov in self.route.active_overlays(body_x, margin=600):
                sx = self._world_to_screen_x(ov.sign_world_x, body_x)
                canvas = self._draw_ortsschild(canvas, ov.sign_text, sx, self._char_sy)

            char = self.renderer.render_character(
                state, self.rig, (self.cfg.width, self.cfg.height)
            )
            frame = Image.alpha_composite(canvas.convert("RGBA"), char)
            frame = self._apply_grade(frame, cel)

            yield frame.convert("RGB")
            frame_i += 1

    def _composite_landscape(self, sky_canvas, state, base):
        from domain.procedural import parallax_offset
        from pathlib import Path

        result = sky_canvas.convert("RGBA")
        w, h = result.size
        for layer in sorted(base, key=lambda b: b.z_index):
            img = self.renderer._svg_to_pil(Path(layer.path))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            ox, oy = parallax_offset(
                state.camera, layer.parallax, state.time, layer.scroll_x, layer.scroll_y
            )
            if layer.repeat_x:
                iw = img.size[0]
                if iw <= 0:
                    continue
                x = (int(ox) % iw) - iw
                while x < w + iw:
                    result.alpha_composite(img, (x, int(oy)))
                    x += iw
            else:
                result.alpha_composite(img, (int(ox), int(oy)))
        return result

    def frames_jpeg(self, quality: int = 75) -> Iterator[bytes]:
        for img in self.frames():
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            yield buf.getvalue()

    def stop(self) -> None:
        self._running = False


def run_mjpeg_server(
    stream: ContinuousWalkStream, host: str = "0.0.0.0", port: int = 8765
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
                "Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}"
            )
            self.send_header("Cache-Control", "no-cache, no-store")
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
    cel = celestial_at(tz=stream.cfg.tz)
    g = scene_grade(cel)
    phases = [
        (0.03, "Neumond"), (0.22, "zunehmende Sichel"), (0.28, "erstes Viertel"),
        (0.47, "zunehmender Mond"), (0.53, "Vollmond"), (0.72, "abnehmender Mond"),
        (0.78, "letztes Viertel"), (0.97, "abnehmende Sichel"), (1.01, "Neumond"),
    ]
    pname = next(n for t, n in phases if cel.moon_phase < t)
    print(f"MJPEG stream → http://{host}:{port}/  (Ctrl+C to stop)")
    print(
        f"Himmel: {cel.local_time.strftime('%Y-%m-%d %H:%M %Z')} · "
        f"Sonne {cel.sun_alt_deg:+.0f}° · Mond {cel.moon_alt_deg:+.0f}° · {pname}"
    )
    print(
        f"Grade: brightness={g.brightness:.2f} saturation={g.saturation:.2f} "
        f"tint=({g.tint_r:+.2f},{g.tint_g:+.2f},{g.tint_b:+.2f})"
    )
    return server


def pipe_to_ffmpeg(
    stream: ContinuousWalkStream, output: str = "pipe:1", extra_args: Sequence[str] | None = None
) -> int:
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(stream.cfg.fps),
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
