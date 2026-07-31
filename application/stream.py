"""Stream: organic features + GeoContext climate + heading sky.

Visual walk cycle stays cartoon-paced; geo distance advances at walk_speed_mps.
Network (weather/OSM) never blocks the frame loop.
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from domain.celestial import heading_project, project_stars
from domain.entities import CharacterRig, FrameState, SceneSpec
from domain.feature_props import FeatureProps, apply_props_to_image
from domain.geo import GeoContext, GeoSample
from domain.geo.route import GeoRoute, demo_route_frankfurt_heidelberg
from domain.procedural import grounded_walk, head_bob, parallax_offset
from domain.sky import celestial_at, scene_grade, sky_colors
from domain.value_objects import Affine, BackgroundLayer, CameraState, Viseme
from domain.zones import LandscapeRoute, RegionMood, generate_route, mood_from_name
from infrastructure.render_cache import (
    cache_stats,
    get_moon_sprite,
    get_sized_rgba,
    get_sky,
    get_svg_rgba,
)
from infrastructure.renderers.pillow_cutout import PillowCutoutRenderer

# Cartoon scroll (world units / s): step_length / (cycle/2) = 40 / 0.3 ≈ 133.3
# Real walking ≈ 1.4 m/s → world_to_meters ≈ 1.4 / 133.3 ≈ 0.0105
DEFAULT_WALK_MPS = 1.4


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
    use_geo: bool = True
    # Real metres advanced per world-unit (auto from walk_speed_mps if None)
    world_to_meters: float | None = None
    walk_speed_mps: float = DEFAULT_WALK_MPS
    geo_live: bool = True
    geo_refresh_s: float = 8.0  # how often to hit weather/OSM (background)


@dataclass(frozen=True, slots=True)
class _DrawItem:
    parallax: float
    path: Path
    start: float
    end: float
    scale: float = 1.0
    y_offset: float = 0.0
    order: int = 0
    props: FeatureProps | None = None


class ContinuousWalkStream:
    def __init__(
        self, scene, renderer, rig, config=None, route=None,
        geo_route: GeoRoute | None = None, geo: GeoContext | None = None,
    ):
        self.scene, self.renderer, self.rig = scene, renderer, rig
        self.cfg = config or StreamConfig(
            fps=float(scene.fps), width=scene.width, height=scene.height,
            scale=getattr(rig, "default_scale", 1.15), character_id=rig.id,
        )
        self.route = route or generate_route(length=8000.0, seed=self.cfg.route_seed)
        self._t0 = time.perf_counter()
        self._running = False
        sample = grounded_walk(0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle)
        self._scroll_speed = sample["scroll_speed"]  # world units / s
        # Map cartoon scroll → real metres/s
        if self.cfg.world_to_meters is None:
            self.cfg.world_to_meters = self.cfg.walk_speed_mps / max(self._scroll_speed, 1e-6)
        self._char_sx = self.cfg.width * 0.40
        self._char_sy = self.cfg.height * 0.82
        tz = ZoneInfo(self.cfg.tz)
        self._clock0 = self.cfg.start_time or datetime.now(tz)
        if self._clock0.tzinfo is None:
            self._clock0 = self._clock0.replace(tzinfo=tz)
        self._last_sky_key = None
        self._last_sky_img = None
        self.geo: GeoContext | None = None
        self._geo_sample: GeoSample | None = None
        self._view_az = 180.0
        self._geo_mood = RegionMood.OFFENLAND
        self._geo_fetching = False
        self._last_geo_fetch = 0.0
        if self.cfg.use_geo:
            try:
                gr = geo_route or (geo.route if geo else demo_route_frankfurt_heidelberg())
                self.geo = geo or GeoContext(gr)
                # elevations once at start (can block briefly — ok before stream)
                try:
                    self.geo.enrich_elevations()
                except Exception:
                    pass
                self.route.mood_provider = self._geo_mood_at
                # prime heading without network
                self._update_heading(0.0)
            except Exception as exc:
                print(f"Geo disabled ({exc})")
                self.geo = None

    def _world_to_geo_m(self, body_x: float) -> float:
        w2m = self.cfg.world_to_meters or 0.0105
        if self.geo is None or self.geo.route.total_m <= 0:
            return body_x * w2m
        return (body_x * w2m) % max(1.0, self.geo.route.total_m)

    def _update_heading(self, body_x: float) -> None:
        """Pure geometry — never blocks."""
        if self.geo is None:
            return
        lon, lat, heading = self.geo.route.sample(self._world_to_geo_m(body_x))
        self._view_az = heading
        # lightweight sample without network for lat/lon display
        if self._geo_sample is None:
            from domain.geo.context import GeoSample
            self._geo_sample = GeoSample(
                distance_m=self._world_to_geo_m(body_x),
                lon=lon, lat=lat, heading_deg=heading,
                elevation_m=self.geo.route.elevation_at(self._world_to_geo_m(body_x)),
                weather=None, landuse=None,
            )
        else:
            # replace with updated pose, keep weather/landuse
            from domain.geo.context import GeoSample
            self._geo_sample = GeoSample(
                distance_m=self._world_to_geo_m(body_x),
                lon=lon, lat=lat, heading_deg=heading,
                elevation_m=self.geo.route.elevation_at(self._world_to_geo_m(body_x)),
                weather=self._geo_sample.weather,
                landuse=self._geo_sample.landuse,
            )

    def _geo_mood_at(self, world_x: float) -> RegionMood | None:
        """Never hits the network — uses last cached mood only."""
        return self._geo_mood

    def _fetch_geo_async(self, body_x: float) -> None:
        """Background weather/OSM update."""
        if self.geo is None or self._geo_fetching or not self.cfg.geo_live:
            return
        now = time.perf_counter()
        if now - self._last_geo_fetch < self.cfg.geo_refresh_s:
            return
        self._geo_fetching = True
        self._last_geo_fetch = now
        dist = self._world_to_geo_m(body_x)

        def worker():
            try:
                s = self.geo.sample(dist, fetch_live=True)
                self._geo_sample = s
                self._view_az = s.heading_deg
                self._geo_mood = mood_from_name(s.mood_name)
            except Exception:
                pass
            finally:
                self._geo_fetching = False

        threading.Thread(target=worker, daemon=True).start()

    def _scene_datetime(self, stream_t):
        return self._clock0 + timedelta(seconds=stream_t * self.cfg.time_scale)

    def _compose_state(self, t):
        gw = grounded_walk(t, step_length=self.cfg.step_length, cycle=self.cfg.cycle, facing=self.cfg.facing)
        bones = dict(gw["bones"])
        bob = head_bob(t, amplitude=2.0, freq=2.5)
        head = bones.get("head", Affine.identity())
        bones["head"] = head.compose(Affine.translate(0.0, bob))
        return FrameState(
            time=t, character_id=self.cfg.character_id, viseme=Viseme.X, jaw_open=0.0,
            bone_transforms=bones, root_position=(self._char_sx, self._char_sy),
            root_rotation_deg=0.0 if self.cfg.facing > 0 else 180.0,
            scale=self.cfg.scale * self.cfg.facing, camera=CameraState(),
        )

    def _world_to_screen_x(self, world_x, body_x, parallax=1.0):
        return self._char_sx + (world_x - body_x) * self.cfg.facing * parallax

    def _make_sky_canvas(self, cel):
        top, bot = sky_colors(cel)
        if self._geo_sample and self._geo_sample.weather:
            cloud = self._geo_sample.weather.sky_cloud_factor
            if cloud > 0.15:
                t = cloud * 0.55
                top = tuple(int(c * (1 - t) + 140 * t) for c in top)
                bot = tuple(int(c * (1 - t) + 160 * t) for c in bot)
        top_q = tuple(c // 4 * 4 for c in top)
        bot_q = tuple(c // 4 * 4 for c in bot)
        key = (self.cfg.width, self.cfg.height, top_q, bot_q)
        if key == self._last_sky_key and self._last_sky_img is not None:
            return self._last_sky_img.copy()
        img = get_sky(self.cfg.width, self.cfg.height, top_q, bot_q)
        self._last_sky_key = key
        self._last_sky_img = img
        return img.copy()

    def _draw_projected_stars(self, canvas, cel):
        if cel.sun_alt_deg > 0:
            return canvas
        strength = min(1.0, max(0.0, (-cel.sun_alt_deg) / 10.0))
        if strength < 0.05:
            return canvas
        lat = self._geo_sample.lat if self._geo_sample else 51.0
        lon = self._geo_sample.lon if self._geo_sample else 10.0
        stars = project_stars(
            cel.local_time, canvas.width, canvas.height,
            lat_deg=lat, lon_deg=lon, view_az_deg=self._view_az, max_mag=2.5,
        )
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for s in stars:
            bright = max(0.15, min(1.0, (2.5 - s.mag) / 3.5))
            r = max(1, int(1 + bright * 2.5))
            alpha = int(255 * strength * bright)
            draw.ellipse([s.x - r, s.y - r, s.x + r, s.y + r], fill=(255, 250, 240, alpha))
        return Image.alpha_composite(canvas, layer)

    def _draw_sun(self, canvas, cel):
        pos = heading_project(
            cel.sun_alt_deg, cel.sun_az_deg, canvas.width, canvas.height, view_az_deg=self._view_az
        )
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

    def _draw_moon(self, canvas, cel):
        pos = heading_project(
            cel.moon_alt_deg, cel.moon_az_deg, canvas.width, canvas.height, view_az_deg=self._view_az
        )
        if pos is None:
            return canvas
        x, y = pos
        r = 22
        moon_img = get_moon_sprite(cel.moon_phase, r)
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(moon_img, (int(x - r - 1), int(y - r - 1)), moon_img)
        return Image.alpha_composite(canvas, layer)

    def _apply_grade(self, img, cel):
        grade = scene_grade(cel)
        brightness, saturation = grade.brightness, grade.saturation
        if self._geo_sample and self._geo_sample.weather:
            w = self._geo_sample.weather
            if w.is_rainy or w.is_foggy:
                brightness *= 0.88
                saturation *= 0.75
            elif w.sky_cloud_factor > 0.6:
                brightness *= 0.94
                saturation *= 0.9
            if w.is_snowy:
                saturation *= 0.7
                brightness *= 1.05
        if abs(brightness - 1.0) < 0.02 and abs(saturation - 1.0) < 0.02 and abs(grade.tint_r) < 0.008:
            return img.convert("RGB")
        rgb = img.convert("RGB")
        if abs(brightness - 1.0) > 0.02:
            rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
        if abs(saturation - 1.0) > 0.02:
            rgb = ImageEnhance.Color(rgb).enhance(saturation)
        if any(abs(v) > 0.008 for v in (grade.tint_r, grade.tint_g, grade.tint_b)):
            import numpy as np
            arr = np.asarray(rgb, dtype=np.float32)
            arr[..., 0] = np.clip(arr[..., 0] + grade.tint_r * 255, 0, 255)
            arr[..., 1] = np.clip(arr[..., 1] + grade.tint_g * 255, 0, 255)
            arr[..., 2] = np.clip(arr[..., 2] + grade.tint_b * 255, 0, 255)
            rgb = Image.fromarray(arr.astype(np.uint8), "RGB")
        return rgb

    def _blit_item(self, canvas, item, body_x):
        if not item.path.is_file():
            return canvas
        left = self._world_to_screen_x(item.start, body_x, item.parallax)
        right = self._world_to_screen_x(item.end, body_x, item.parallax)
        if self.cfg.facing < 0:
            left, right = right, left
        if right < -60 or left > self.cfg.width + 60:
            return canvas
        panel_w = max(1, int(abs(right - left)))
        panel_h = max(1, int(self.cfg.height * item.scale))
        scaled = get_sized_rgba(
            item.path, item.props, panel_w, panel_h,
            self.renderer._svg_to_pil, apply_props_to_image,
        )
        y = int(self.cfg.height - panel_h + item.y_offset)
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        canvas.alpha_composite(scaled, (int(left), y))
        return canvas

    def _blit_tiled_layer(self, canvas, layer, state):
        path = Path(layer.path)
        if not path.is_file():
            return canvas
        img = get_svg_rgba(path, self.renderer._svg_to_pil)
        ox, oy = parallax_offset(state.camera, layer.parallax, state.time, layer.scroll_x, layer.scroll_y)
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        w = canvas.size[0]
        if layer.repeat_x:
            iw = img.size[0]
            if iw <= 0:
                return canvas
            x = (int(ox) % iw) - iw
            while x < w + iw:
                canvas.alpha_composite(img, (x, int(oy)))
                x += iw
        else:
            canvas.alpha_composite(img, (int(ox), int(oy)))
        return canvas

    def _collect_draw_items(self, body_x):
        return [
            _DrawItem(
                parallax=f.parallax, path=self.route.feature_object_path(f.kind),
                start=f.start, end=f.end, scale=f.scale, y_offset=f.y_offset, props=f.props,
            )
            for f in self.route.active_features(body_x)
        ]

    def _draw_ortsschild(self, img, text, screen_x, ground_y):
        if not text or screen_x < -100 or screen_x > img.width + 100:
            return img
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        w, h, cx = 150, 48, int(screen_x)
        post_top = ground_y - 50
        plate_top = post_top - h + 8
        draw.rectangle([cx - 4, int(post_top), cx + 4, int(ground_y)], fill=(80, 80, 80, 255))
        x0, y0 = cx - w // 2, int(plate_top)
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=4,
            fill=(245, 245, 245, 255), outline=(25, 25, 25, 255), width=3)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, y0 + (h - th) // 2 - 1), text, fill=(20, 20, 20, 255), font=font)
        return Image.alpha_composite(img.convert("RGBA"), layer)

    def frames(self):
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0
        base_layers = sorted(
            self.route.base_layers(self._scroll_speed, self.cfg.facing), key=lambda L: L.parallax
        )
        w2m = self.cfg.world_to_meters or 0.0105
        print(
            f"Walk: {self._scroll_speed:.0f} wu/s visual · "
            f"geo {self.cfg.walk_speed_mps:.1f} m/s "
            f"(world_to_m={w2m:.4f})"
        )
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
            # cheap every frame: heading from polyline
            if frame_i % 6 == 0:
                self._update_heading(body_x)
            # network only in background, rarely
            if frame_i % max(1, int(self.cfg.fps * 2)) == 0:
                self._fetch_geo_async(body_x)
            self.route.ensure_ahead(body_x, look_ahead=12000.0)
            if frame_i % 48 == 0:
                self.route.prune_behind(body_x, keep_behind=10000.0)
            cel = celestial_at(self._scene_datetime(t), tz=self.cfg.tz)
            canvas = self._make_sky_canvas(cel)
            canvas = self._draw_projected_stars(canvas, cel)
            canvas = self._draw_moon(canvas, cel)
            canvas = self._draw_sun(canvas, cel)
            items = sorted(self._collect_draw_items(body_x), key=lambda it: (it.parallax, it.order))
            bi = 0
            for item in items:
                while bi < len(base_layers) and base_layers[bi].parallax <= item.parallax:
                    canvas = self._blit_tiled_layer(canvas, base_layers[bi], state)
                    bi += 1
                canvas = self._blit_item(canvas, item, body_x)
            while bi < len(base_layers):
                canvas = self._blit_tiled_layer(canvas, base_layers[bi], state)
                bi += 1
            for reg in self.route.active_regions(body_x, margin=600):
                if reg.sign_text:
                    sx = self._world_to_screen_x(reg.sign_world_x, body_x, parallax=1.0)
                    canvas = self._draw_ortsschild(canvas, reg.sign_text, sx, self._char_sy)
            char = self.renderer.render_character(state, self.rig, (self.cfg.width, self.cfg.height))
            frame = Image.alpha_composite(canvas.convert("RGBA"), char)
            yield self._apply_grade(frame, cel)
            frame_i += 1

    def frames_jpeg(self, quality=75):
        for img in self.frames():
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            yield buf.getvalue()

    def stop(self):
        self._running = False


def run_mjpeg_server(stream, host="0.0.0.0", port=8765):
    boundary = b"frame"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = (
                    b"<!DOCTYPE html><html><body style='margin:0;background:#111;"
                    b"display:flex;justify-content:center;align-items:center;height:100vh'>"
                    b"<img src='/stream.mjpg'></body></html>"
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
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}")
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
    geo_info = "off"
    if stream.geo is not None:
        geo_info = f"on · {stream.geo.route.total_m/1000:.1f} km · {stream.cfg.walk_speed_mps:.1f} m/s"
    print(f"MJPEG → http://{host}:{port}/ · Features={len(stream.route.features)} · Geo={geo_info}")
    return server


def pipe_to_ffmpeg(stream, output="pipe:1", extra_args=None):
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
