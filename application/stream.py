"""Continuous real-time stream with German landscape zones."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator, Sequence

from PIL import Image, ImageDraw, ImageFont

from domain.entities import CharacterRig, FrameState, SceneSpec
from domain.procedural import grounded_walk, head_bob
from domain.value_objects import Affine, CameraState, Viseme
from domain.zones import ZoneSequence, default_german_tour
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
    fade_s: float = 1.2


class ContinuousWalkStream:
    """Live walk + zone landscapes + Ortsschild transitions."""

    def __init__(
        self,
        scene: SceneSpec,
        renderer: PillowCutoutRenderer,
        rig: CharacterRig,
        config: StreamConfig | None = None,
        zones: ZoneSequence | None = None,
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
        self.zones = zones or default_german_tour()
        self._t0 = time.perf_counter()
        self._running = False
        sample = grounded_walk(
            0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle
        )
        self._scroll_speed = sample["scroll_speed"]

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

    def _blend_images(self, a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        t = max(0.0, min(1.0, t))
        return Image.blend(a.convert("RGB"), b.convert("RGB"), t)

    def _draw_ortsschild(
        self, img: Image.Image, text: str, alpha: float = 1.0
    ) -> Image.Image:
        if alpha <= 0.01:
            return img
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cx, cy = int(img.width * 0.72), int(img.height * 0.55)
        w, h = 160, 56
        x0, y0 = cx - w // 2, cy - h // 2
        draw.rectangle(
            [cx - 4, cy + h // 2 - 4, cx + 4, cy + h // 2 + 40],
            fill=(90, 90, 90, 230),
        )
        draw.rounded_rectangle(
            [x0, y0, x0 + w, y0 + h],
            radius=4,
            fill=(245, 245, 245, 240),
            outline=(30, 30, 30, 255),
            width=3,
        )
        draw.rounded_rectangle(
            [x0 + 5, y0 + 5, x0 + w - 5, y0 + h - 5],
            radius=2,
            outline=(30, 30, 30, 200),
            width=1,
        )
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, cy - th // 2 - 2), text, fill=(20, 20, 20, 255), font=font
        )
        if alpha < 1.0:
            r, g, b, a = overlay.split()
            a = a.point(lambda p: int(p * alpha))
            overlay = Image.merge("RGBA", (r, g, b, a))
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def frames(self) -> Iterator[Image.Image]:
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0
        facing = self.cfg.facing

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
            distance = self._scroll_speed * t

            cur, nxt, blend = self.zones.active_blend(
                distance, self._scroll_speed, self.cfg.fade_s
            )
            layers_a = self.zones.layers_for(cur.zone, self._scroll_speed, facing)
            img = self.renderer.render_image(
                state,
                self.rig,
                (self.cfg.width, self.cfg.height),
                backgrounds=layers_a,
            )

            if nxt is not None and blend > 0.0:
                layers_b = self.zones.layers_for(nxt.zone, self._scroll_speed, facing)
                img_b = self.renderer.render_image(
                    state,
                    self.rig,
                    (self.cfg.width, self.cfg.height),
                    backgrounds=layers_b,
                )
                img = self._blend_images(img, img_b, blend)

            for ev in self.zones.transitions():
                window = self._scroll_speed * 2.5
                d = abs(distance - ev.at_distance)
                if d < window and ev.show_sign:
                    alpha = max(0.0, 1.0 - d / window)
                    img = self._draw_ortsschild(img, ev.sign_text, alpha=alpha)
                    break

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
    print(f"MJPEG stream → http://{host}:{port}/  (Ctrl+C to stop)")
    print("Zones: Felder → Landstraße → Musterdorf → Stadtwald → Neustadt → Felder…")
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
