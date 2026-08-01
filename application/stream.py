"""Stream: organic features + GeoContext + VG250 Gemeinde Ortsschilder.

Draw order: mid scenery → ALL features (behind road) → road → signs → character.
Feature content underside on landscape ground plane (no transparent pad).
"""
from __future__ import annotations

import io
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

from domain.celestial import celestial_at, sky_colors
from domain.geo.context import GeoContext
from domain.procedural import grounded_walk, head_bob, parallax_offset
from domain.rig import load_character_rig
from domain.value_objects import Affine, BackgroundLayer, CameraState, FrameState, Viseme
from domain.zones import LandscapeRoute, RegionMood, mood_from_name
from infrastructure.render_cache import apply_props_to_image, get_sized_rgba, get_svg_rgba
from infrastructure.renderers import get_renderer
from PIL import Image, ImageDraw, ImageFont


@dataclass
class StreamConfig:
    width: int = 960
    height: int = 540
    fps: int = 24
    duration: float | None = None
    scroll_speed: float = 80.0
    walk_speed_mps: float = 1.4
    step_length: float = 28.0
    cycle: float = 0.85
    facing: float = 1.0
    scale: float = 1.15
    character_id: str = "walker"
    geo_live: bool = True
    geo_refresh_s: float = 8.0
    time_scale: float = 1.0
    tz: str = "Europe/Berlin"


@dataclass
class _DrawItem:
    parallax: float
    path: Path
    start: float
    end: float
    scale: float = 1.0
    y_offset: float = 0.0
    props: object = None
    order: int = 0


class ContinuousWalkStream:
    """Main MJPEG walk stream with geo-driven landscape."""

    def __init__(self, cfg: StreamConfig | None = None, rig=None, route=None, geo=None):
        self.cfg = cfg or StreamConfig()
        self.rig = rig or load_character_rig(self.cfg.character_id)
        self.route = route or LandscapeRoute(seed=42)
        self.geo = geo
        self.renderer = get_renderer()
        self._running = False
        self._t0 = 0.0
        self._scroll_speed = self.cfg.scroll_speed
        self._char_sx = int(self.cfg.width * 0.35)
        self._char_sy = int(self.cfg.height * 0.72)
        self._geo_sample = None
        self._geo_mood = RegionMood.OFFENLAND
        self._geo_fetching = False
        self._last_geo_fetch = 0.0
        self._curve = 0.0
        self._curve_back = 0.0
        self._view_az = 0.0
        self._clock0 = datetime.now(ZoneInfo(self.cfg.tz))
        self._base_layers = None
        self._base_layers_mood = None

    def _world_to_geo_m(self, body_x):
        return body_x * float(self.cfg.walk_speed_mps) / max(1.0, float(self._scroll_speed))

    def _fetch_geo_async(self, body_x):
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
                self._geo_mood = mood_from_name(s.mood_name)
                if s.landuse is not None:
                    lu = s.landuse
                    self.route._building_density = float(getattr(lu, "building_density", 0) or 0)
                    self.route._forest_density = float(getattr(lu, "forest_density", 0) or 0)
                    self.route._farm_density = float(getattr(lu, "farm_density", 0) or 0)
                    self.route._industrial_density = float(getattr(lu, "industrial_density", 0) or 0)
                    if hasattr(lu, "building_scale_mul"):
                        self.route._building_height_scale = float(lu.building_scale_mul())
                        self.route._skyline_scale = float(lu.skyline_scale_mul())
                if getattr(s, "water", None) is not None:
                    self._apply_water_climate(s, dist)
            except Exception:
                pass
            finally:
                self._geo_fetching = False

        threading.Thread(target=worker, daemon=True).start()

    def _apply_water_climate(self, sample, dist_m: float) -> None:
        """OSM water → bridge on crossing, river/lake in background."""
        w = getattr(sample, "water", None)
        if w is None:
            return
        from domain.geo.water import WaterHit
        scale = float(self._scroll_speed) / max(0.1, float(self.cfg.walk_speed_mps))
        self.route._geo_world_scale = scale
        hits = []
        if float(getattr(w, "cross", 0.0) or 0.0) >= 0.30:
            hits.append(WaterHit(dist_m, getattr(w, "kind", None) or "river", "cross", float(getattr(w, "width_hint_m", 0) or 30.0)))
            hits.append(WaterHit(dist_m, getattr(w, "kind", None) or "river", "left", max(40.0, float(getattr(w, "width_hint_m", 0) or 30.0) * 1.2)))
        else:
            bg = max(float(getattr(w, "left", 0) or 0), float(getattr(w, "right", 0) or 0))
            if bg >= 0.22:
                hits.append(WaterHit(dist_m, getattr(w, "kind", None) or "water", "left", float(getattr(w, "width_hint_m", 0) or 40.0)))
        if hits and hasattr(self.route, "inject_water_hits"):
            self.route.inject_water_hits(hits)

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
        canvas = Image.new("RGB", (self.cfg.width, self.cfg.height))
        draw = ImageDraw.Draw(canvas)
        for y in range(self.cfg.height):
            t = y / max(1, self.cfg.height - 1)
            r = int(top[0] + (bot[0] - top[0]) * t)
            g = int(top[1] + (bot[1] - top[1]) * t)
            b = int(top[2] + (bot[2] - top[2]) * t)
            draw.line([(0, y), (self.cfg.width, y)], fill=(r, g, b))
        return canvas.convert("RGBA")

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
        path_l = str(item.path).lower()
        is_gc = any(path_l.endswith(n) for n in ("acker.svg", "sumpf.svg", "fluss.svg", "see.svg"))
        if is_gc:
            panel_h = min(panel_h, max(24, int(self.cfg.height * 0.75) - int(self.cfg.height * 0.58)))
        if path_l.endswith("bruecke.svg"):
            panel_h = min(panel_h, max(60, int(self.cfg.height * 0.22)))
        scaled = get_sized_rgba(item.path, item.props, panel_w, panel_h, self.renderer._svg_to_pil, apply_props_to_image)
        sw, sh = scaled.size
        ground_y = int(self.cfg.height * 0.75)
        y = int(ground_y - sh + item.y_offset)
        if is_gc:
            y = max(y, int(self.cfg.height * 0.58))
        x = int(left + (panel_w - sw) * 0.5)
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        canvas.alpha_composite(scaled, (x, y))
        return canvas

    def _draw_boundary_signs(self, canvas, body_x):
        return canvas

    def _apply_grade(self, frame, cel):
        return frame.convert("RGB")

    def _update_heading(self, body_x):
        pass

    def _smooth_view(self, dt):
        pass

    def _draw_projected_stars(self, canvas, cel):
        return canvas

    def _draw_moon(self, canvas, cel):
        return canvas

    def _draw_sun(self, canvas, cel):
        return canvas

    def _scene_datetime(self, stream_t):
        return self._clock0 + timedelta(seconds=stream_t * self.cfg.time_scale)

    def frames(self):
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0
        self._base_layers = None
        self._base_layers_mood = None
        print(f"Walk: {self._scroll_speed:.0f} wu/s")
        while self._running:
            t = frame_i * dt
            if self.cfg.duration is not None and t >= self.cfg.duration:
                break
            sleep = self._t0 + frame_i * dt - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            state = self._compose_state(t)
            body_x = self._scroll_speed * t
            if frame_i % max(1, int(self.cfg.fps * 2)) == 0:
                self._fetch_geo_async(body_x)
            mood = self._geo_mood or RegionMood.OFFENLAND
            if self._base_layers is None or self._base_layers_mood != mood:
                try:
                    self._base_layers = sorted(
                        self.route.base_layers(self._scroll_speed, self.cfg.facing, mood=mood),
                        key=lambda L: L.parallax,
                    )
                except TypeError:
                    self._base_layers = sorted(
                        self.route.base_layers(self._scroll_speed, self.cfg.facing),
                        key=lambda L: L.parallax,
                    )
                self._base_layers_mood = mood
            self.route.ensure_ahead(body_x, look_ahead=12000.0)
            if frame_i % 48 == 0:
                self.route.prune_behind(body_x, keep_behind=10000.0)
            cel = celestial_at(self._scene_datetime(t), tz=self.cfg.tz)
            canvas = self._make_sky_canvas(cel)
            base_layers = self._base_layers or []
            for layer in [L for L in base_layers if L.parallax < 0.99]:
                canvas = self._blit_tiled_layer(canvas, layer, state)
            for item in sorted(self._collect_draw_items(body_x), key=lambda it: (it.parallax, it.order)):
                canvas = self._blit_item(canvas, item, body_x)
            for layer in [L for L in base_layers if L.parallax >= 0.99]:
                canvas = self._blit_tiled_layer(canvas, layer, state)
            canvas = self._draw_boundary_signs(canvas, body_x)
            char = self.renderer.render_character(state, self.rig, (self.cfg.width, self.cfg.height))
            yield self._apply_grade(Image.alpha_composite(canvas.convert("RGBA"), char), cel)
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
                    b"<img src='/stream.mjpg' style='max-width:100%;max-height:100%'/></body></html>"
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
                "multipart/x-mixed-replace; boundary=" + boundary.decode(),
            )
            self.end_headers()
            try:
                for jpeg in stream.frames_jpeg():
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = HTTPServer((host, port), Handler)
    print(f"MJPEG http://{host}:{port}/stream.mjpg")
    try:
        server.serve_forever()
    finally:
        stream.stop()
        server.server_close()
