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
    get_moon_sprite,
    get_sized_rgba,
    get_sky,
    get_svg_rgba,
)
from infrastructure.renderers.pillow_cutout import PillowCutoutRenderer

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
    world_to_meters: float | None = None
    walk_speed_mps: float = DEFAULT_WALK_MPS
    geo_live: bool = True
    geo_refresh_s: float = 8.0


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
    def __init__(self, scene, renderer, rig, config=None, route=None, geo_route=None, geo=None):
        self.scene, self.renderer, self.rig = scene, renderer, rig
        self.cfg = config or StreamConfig(
            fps=float(scene.fps), width=scene.width, height=scene.height,
            scale=getattr(rig, "default_scale", 1.15), character_id=rig.id,
        )
        self.route = route or generate_route(length=8000.0, seed=self.cfg.route_seed)
        self._t0 = time.perf_counter()
        self._running = False
        sample = grounded_walk(0.0, step_length=self.cfg.step_length, cycle=self.cfg.cycle)
        self._scroll_speed = sample["scroll_speed"]
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
        self.geo = None
        self._geo_sample = None
        self._view_az = 90.0
        self._view_az_target = 90.0
        self._heading_s = 180.0
        self._curve = 0.0
        self._curve_sky = 0.0
        self._curve_back = 0.0
        self._curve_target = 0.0
        self._turning = False
        self._curve_hist = deque(maxlen=200)
        self._curve_hist_t = deque(maxlen=200)
        self._geo_mood = RegionMood.OFFENLAND
        self._geo_fetching = False
        self._last_geo_fetch = 0.0
        self._place_cache = {}
        self._boundary_segments = []
        self._boundary_lock = threading.Lock()
        if self.cfg.use_geo:
            try:
                gr = geo_route or (geo.route if geo else demo_route_frankfurt_heidelberg())
                self.geo = geo or GeoContext(gr)
                try:
                    self.geo.enrich_elevations()
                except Exception:
                    pass
                self.route.mood_provider = self._geo_mood_at
                self.route.place_name_provider = self._place_name_at
                self._update_heading(0.0)
                self._start_boundary_preload(gr)
            except Exception as exc:
                print(f"Geo disabled ({exc})")
                self.geo = None

    def _world_to_geo_m(self, body_x):
        w2m = self.cfg.world_to_meters or 0.0105
        if self.geo is None or self.geo.route.total_m <= 0:
            return body_x * w2m
        return (body_x * w2m) % max(1.0, self.geo.route.total_m)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return ((b - a + 540.0) % 360.0) - 180.0

    @staticmethod
    def _angle_lerp(a: float, b: float, t: float) -> float:
        return (a + ContinuousWalkStream._angle_diff(a, b) * t) % 360.0

    def _update_heading(self, body_x):
        if self.geo is None:
            return
        cur_m = self._world_to_geo_m(body_x)
        lon, lat, heading_raw = self.geo.route.sample(cur_m)
        self._heading_s = self._angle_lerp(self._heading_s, heading_raw, 0.4)
        heading = self._heading_s
        total = max(1.0, float(self.geo.route.total_m))
        look_m = float(self.cfg.walk_speed_mps or 1.4) * 3.0
        ahead_m = (cur_m + look_m) % total
        try:
            _, _, heading_ahead = self.geo.route.sample(ahead_m)
        except Exception:
            heading_ahead = heading_raw
        self._view_az_target = (heading_ahead - 90.0) % 360.0
        from domain.geo.context import GeoSample as GS
        prev = self._geo_sample
        self._geo_sample = GS(
            distance_m=cur_m, lon=lon, lat=lat, heading_deg=heading,
            elevation_m=self.geo.route.elevation_at(cur_m),
            weather=prev.weather if prev else None,
            landuse=prev.landuse if prev else None,
        )

    def _curve_at_delay(self, delay_s: float) -> float:
        if not self._curve_hist_t:
            return 0.0
        target = time.perf_counter() - delay_s
        best = 0.0
        for t, c in zip(self._curve_hist_t, self._curve_hist):
            if t <= target:
                best = c
            else:
                break
        return best

    def _smooth_view(self, dt: float) -> None:
        error = self._angle_diff(self._view_az, self._view_az_target)
        if abs(error) > 0.1:
            a = 1.0 - math.exp(-dt / 1.2)
            self._view_az = self._angle_lerp(self._view_az, self._view_az_target, a)
        else:
            self._view_az = self._view_az_target
        MAX_BEND = 1.0
        raw = error / 15.0
        target_curve = max(-MAX_BEND, min(MAX_BEND, raw))
        if abs(target_curve) < 0.04:
            target_curve = 0.0
        a = 1.0 - math.exp(-dt / 0.25)
        self._curve += (target_curve - self._curve) * a
        if abs(self._curve) < 0.02:
            self._curve = 0.0
        now = time.perf_counter()
        self._curve_hist.append(self._curve)
        self._curve_hist_t.append(now)
        self._curve_sky = self._curve_at_delay(3.0)
        self._curve_back = self._curve_at_delay(4.0)

    def _geo_mood_at(self, world_x):
        return self._geo_mood

    def _place_name_at(self, world_x):
        if self.geo is None:
            return None
        key = int(self._world_to_geo_m(world_x) // 500)
        if key in self._place_cache:
            return self._place_cache[key]
        try:
            lon, lat, _ = self.geo.route.sample(self._world_to_geo_m(world_x))
            try:
                from domain.geo.boundaries import gemeinde_name
                name = gemeinde_name(lon, lat)
            except Exception:
                name = None
            if not name:
                from domain.geo.places import reverse_place_name
                name = reverse_place_name(lat, lon)
            if name:
                self._place_cache[key] = name
            return name
        except Exception:
            return None

    def _start_boundary_preload(self, gr) -> None:
        def worker():
            try:
                from domain.geo.boundaries import ATTRIBUTION_SHORT, segments_along_route
                samples = []
                step = 400.0
                d = 0.0
                total = max(1.0, float(gr.total_m))
                while d <= total:
                    lon, lat, _ = gr.sample(d)
                    samples.append((lon, lat, d))
                    d += step
                segs = segments_along_route(samples, min_length_m=1500.0, timeout=20.0)
                with self._boundary_lock:
                    self._boundary_segments = segs
                print(f"Ortsschilder: {len(segs)} Gemeinde-Abschnitte (VG250) · {ATTRIBUTION_SHORT}")
            except Exception as e:
                print(f"Boundary preload skipped ({e})")
        threading.Thread(target=worker, daemon=True).start()

    def _geo_m_to_screen_x(self, target_m: float, cur_m: float) -> float:
        total = float(self.geo.route.total_m) if self.geo is not None else 0.0
        dm = target_m - cur_m
        if total > 1.0 and abs(dm) > total * 0.5:
            dm -= math.copysign(total, dm)
        w2m = self.cfg.world_to_meters or 0.0105
        return self._char_sx + (dm / max(w2m, 1e-9)) * self.cfg.facing

    def _draw_boundary_signs(self, canvas, body_x):
        if self.geo is None:
            return canvas
        with self._boundary_lock:
            segs = list(self._boundary_segments)
        if not segs:
            return canvas
        cur_m = self._world_to_geo_m(body_x)
        total = float(self.geo.route.total_m) or 1e18
        for seg in segs:
            for mark_m, is_exit in ((seg.enter_m, False), (seg.exit_m, True)):
                dm = mark_m - cur_m
                if total > 1.0 and abs(dm) > total * 0.5:
                    dm -= math.copysign(total, dm)
                if abs(dm) > 2000.0:
                    continue
                sx = self._geo_m_to_screen_x(mark_m, cur_m)
                canvas = self._draw_ortsschild(canvas, seg.name, sx, self._char_sy, exit_sign=is_exit)
        return canvas

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
        """OSM water → bridge on crossing, river/lake in background (both sides)."""
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
        if hits:
            self.route.inject_water_hits(hits)

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
        path_l = str(item.path).lower()
        is_ground_cover = path_l.endswith("acker.svg") or path_l.endswith("sumpf.svg")
        if is_ground_cover:
            max_h = max(24, int(self.cfg.height * 0.75) - int(self.cfg.height * 0.58))
            panel_h = min(panel_h, max_h)
        scaled = get_sized_rgba(
            item.path, item.props, panel_w, panel_h,
            self.renderer._svg_to_pil, apply_props_to_image,
        )
        sw, sh = scaled.size
        ground_y = int(self.cfg.height * 0.75)
        y = int(ground_y - sh + item.y_offset)
        if is_ground_cover:
            opaque_top = int(self.cfg.height * 0.58)
            if y < opaque_top:
                y = opaque_top
        x = int(left + (panel_w - sw) * 0.5)
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        canvas.alpha_composite(scaled, (x, y))
        return canvas

    def _is_road_layer(self, layer) -> bool:
        p = str(getattr(layer, "path", "")).lower()
        if "landstrasse" in p or "ground" in p:
            return True
        return abs(float(getattr(layer, "parallax", 0.0)) - 1.0) < 0.05

    def _warp_road_layer(self, layer_img):
        kf = self._curve
        kb = self._curve_back
        if abs(kf) < 0.02 and abs(kb) < 0.02:
            return layer_img
        try:
            import numpy as np
        except ImportError:
            return layer_img
        if layer_img.mode != "RGBA":
            layer_img = layer_img.convert("RGBA")
        arr = np.asarray(layer_img)
        h, w = arr.shape[:2]
        max_dy_f = float(max(-28.0, min(28.0, kf * 28.0)))
        max_dy_b = float(max(-28.0, min(28.0, kb * 28.0)))
        cx = float(self._char_sx)
        facing = 1.0 if self.cfg.facing >= 0 else -1.0
        out = np.zeros_like(arr)
        xs = np.arange(w, dtype=np.float32)
        along = (xs - cx) * facing
        half = max(1.0, w * 0.5)
        t = along / half
        ahead_w = np.clip(t, 0.0, None)
        behind_w = np.clip(-t, 0.0, None)
        af = ahead_w * ahead_w
        bf = behind_w * behind_w
        dy = (max_dy_f * af + max_dy_b * bf).astype(np.int32)
        for x in range(w):
            d = int(dy[x])
            if d == 0:
                out[:, x] = arr[:, x]
                continue
            src_y = np.arange(h, dtype=np.int32) - d
            valid = (src_y >= 0) & (src_y < h)
            out[valid, x] = arr[src_y[valid], x]
        return Image.fromarray(out, "RGBA")

    def _blit_tiled_layer(self, canvas, layer, state):
        path = Path(layer.path)
        if not path.is_file():
            return canvas
        img = get_svg_rgba(path, self.renderer._svg_to_pil)
        ox, oy = parallax_offset(state.camera, layer.parallax, state.time, layer.scroll_x, layer.scroll_y)
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        w = canvas.size[0]
        bend = self._is_road_layer(layer) and (abs(self._curve) >= 0.02 or abs(self._curve_back) >= 0.02)
        if bend:
            temp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            if layer.repeat_x:
                iw = img.size[0]
                if iw <= 0:
                    return canvas
                x = (int(ox) % iw) - iw
                while x < w + iw:
                    temp.alpha_composite(img, (x, int(oy)))
                    x += iw
            else:
                temp.alpha_composite(img, (int(ox), int(oy)))
            temp = self._warp_road_layer(temp)
            canvas.alpha_composite(temp)
            return canvas
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

    def _draw_ortsschild(self, img, text, screen_x, ground_y, *, exit_sign=False):
        try:
            from domain.signs import draw_ortsschild as _draw
            return _draw(img, text, screen_x, ground_y, exit_sign=exit_sign)
        except Exception:
            pass
        if not text or screen_x < -120 or screen_x > img.width + 120:
            return img
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w, h = max(120, tw + 36), max(42, th + 20)
        cx = int(screen_x)
        post_top = ground_y - 52
        y0 = int(post_top - h + 6)
        x0 = cx - w // 2
        draw.rectangle([cx - 3, int(post_top), cx + 3, int(ground_y)], fill=(90, 90, 90, 255))
        yellow, black = (255, 204, 0, 255), (20, 20, 20, 255)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=yellow, outline=black, width=3)
        draw.rectangle([x0 + 3, y0 + 3, x0 + w - 3, y0 + h - 3], outline=black, width=1)
        draw.text((cx - tw // 2, y0 + (h - th) // 2 - 1), text, fill=black, font=font)
        if exit_sign:
            draw.line([(x0 + 6, y0 + h - 6), (x0 + w - 6, y0 + 6)], fill=black, width=4)
        return Image.alpha_composite(img.convert("RGBA"), layer)

    def frames(self):
        self._running = True
        self._t0 = time.perf_counter()
        dt = 1.0 / max(self.cfg.fps, 1.0)
        frame_i = 0
        self._base_layers = None
        self._base_layers_mood = None
        print(f"Walk: {self._scroll_speed:.0f} wu/s · geo {self.cfg.walk_speed_mps:.1f} m/s")
        while self._running:
            t = frame_i * dt
            if self.cfg.duration is not None and t >= self.cfg.duration:
                break
            sleep = self._t0 + frame_i * dt - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            state = self._compose_state(t)
            body_x = self._scroll_speed * t
            self._update_heading(body_x)
            self._smooth_view(dt)
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
            mood = self._geo_mood if hasattr(self, "_geo_mood") else None
            try:
                from domain.zones import RegionMood
                mood = mood or RegionMood.OFFENLAND
            except Exception:
                mood = None
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
            base_layers = self._base_layers or []
            mid_layers = [L for L in base_layers if L.parallax < 0.99]
            road_layers = [L for L in base_layers if L.parallax >= 0.99]
            for layer in mid_layers:
                canvas = self._blit_tiled_layer(canvas, layer, state)
            items = sorted(self._collect_draw_items(body_x), key=lambda it: (it.parallax, it.order))
            for item in items:
                canvas = self._blit_item(canvas, item, body_x)
            for layer in road_layers:
                canvas = self._blit_tiled_layer(canvas, layer, state)
            canvas = self._draw_boundary_signs(canvas, body_x)
            char = self.renderer.render_character(
                state, self.rig, (self.cfg.width, self.cfg.height)
            )
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
    try:
        from domain.geo.boundaries import ATTRIBUTION_SHORT
        attr = ATTRIBUTION_SHORT
    except Exception:
        attr = ""
    print(f"MJPEG -> http://{host}:{port}/ · Features={len(stream.route.features)} · Geo={geo_info}")
    if attr:
        print(f"  Daten: {attr}")
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
