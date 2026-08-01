"""Organic infinite landscape with stable real-map Ortsschilder.

Ortsschilder: only real place names; consecutive same-name regions form one
stretch with a single Eingang + Ausgang (MIN_PLACE_STRETCH / MIN_PLACE_GAP).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Callable

from .feature_props import FeatureProps, sample_props
from .value_objects import BackgroundLayer


class Tag(Flag):
    NATURE = auto()
    FARM = auto()
    RURAL = auto()
    FOREST = auto()
    URBAN = auto()
    INDUSTRIAL = auto()
    RUIN = auto()
    ROAD = auto()


class FeatureKind(str, Enum):
    ACKER = "acker"
    TIERE = "tiere"
    FELSEN = "felsen"
    INDUSTRIE = "industrie"
    RUINE = "ruine"
    WRACK = "wrack"
    BAUERNHOF = "bauernhof"
    SUMPF = "sumpf"
    WINDRAD = "windrad"
    HOCHSPANNUNG = "hochspannung"
    KAPELLE = "kapelle"
    HEUBALLEN = "heuballen"
    HAUS = "haus"
    BAUM = "baum"
    LATERNE = "laterne"
    LAGERHALLE = "lagerhalle"
    BUSCH = "busch"
    BRUECKE = "bruecke"
    FLUSS = "fluss"
    SEE = "see"


class Depth(int, Enum):
    NEAR = 0
    MID = 1
    FAR = 2


class RegionMood(str, Enum):
    OFFENLAND = "offenland"
    DORF = "dorf"
    STADT = "stadt"
    WALD = "wald"


NEAR_PARALLAX = 0.92
FAR_PARALLAX = 0.28
NEAR_SCALE = 1.00
FAR_SCALE = 0.45
NEAR_Y = 0.0
FAR_Y = -70.0

DEPTH_T_RANGE = {
    Depth.NEAR: (0.00, 0.33),
    Depth.MID: (0.33, 0.66),
    Depth.FAR: (0.66, 1.00),
}


def _lerp(a, b, t):
    return a + (b - a) * t


def depth_from_t(t):
    t = max(0.0, min(1.0, t))
    return (
        _lerp(NEAR_PARALLAX, FAR_PARALLAX, t),
        _lerp(NEAR_SCALE, FAR_SCALE, t),
        _lerp(NEAR_Y, FAR_Y, t),
    )

FEATURE_TAGS = {
    FeatureKind.ACKER: Tag.FARM | Tag.RURAL | Tag.NATURE,
    FeatureKind.TIERE: Tag.FARM | Tag.RURAL,
    FeatureKind.FELSEN: Tag.NATURE | Tag.FOREST,
    FeatureKind.INDUSTRIE: Tag.INDUSTRIAL | Tag.URBAN,
    FeatureKind.RUINE: Tag.RUIN | Tag.RURAL,
    FeatureKind.WRACK: Tag.ROAD | Tag.RUIN,
    FeatureKind.BAUERNHOF: Tag.FARM | Tag.RURAL,
    FeatureKind.SUMPF: Tag.NATURE | Tag.FOREST,
    FeatureKind.WINDRAD: Tag.INDUSTRIAL | Tag.RURAL,
    FeatureKind.HOCHSPANNUNG: Tag.INDUSTRIAL | Tag.ROAD,
    FeatureKind.KAPELLE: Tag.RURAL | Tag.FARM,
    FeatureKind.HEUBALLEN: Tag.FARM | Tag.RURAL,
    FeatureKind.HAUS: Tag.URBAN | Tag.RURAL,
    FeatureKind.BAUM: Tag.FOREST | Tag.NATURE,
    FeatureKind.LATERNE: Tag.URBAN | Tag.ROAD,
    FeatureKind.LAGERHALLE: Tag.INDUSTRIAL | Tag.URBAN,
    FeatureKind.BUSCH: Tag.NATURE | Tag.FOREST | Tag.RURAL,
    FeatureKind.BRUECKE: Tag.ROAD | Tag.URBAN,
    FeatureKind.FLUSS: Tag.NATURE | Tag.FOREST,
    FeatureKind.SEE: Tag.NATURE | Tag.FOREST,
}

BASE_WEIGHTS = {
    FeatureKind.ACKER: 1.3, FeatureKind.TIERE: 0.9, FeatureKind.FELSEN: 0.7,
    FeatureKind.INDUSTRIE: 0.15, FeatureKind.RUINE: 0.25, FeatureKind.WRACK: 0.2,
    FeatureKind.BAUERNHOF: 0.5, FeatureKind.SUMPF: 0.3, FeatureKind.WINDRAD: 0.35,
    FeatureKind.HOCHSPANNUNG: 0.3, FeatureKind.KAPELLE: 0.25, FeatureKind.HEUBALLEN: 1.0,
    FeatureKind.HAUS: 0.2, FeatureKind.BAUM: 0.8, FeatureKind.LATERNE: 0.1,
    FeatureKind.LAGERHALLE: 0.1, FeatureKind.BUSCH: 0.9,
    FeatureKind.BRUECKE: 0.0, FeatureKind.FLUSS: 0.0, FeatureKind.SEE: 0.0,
}

MOOD_TAG_BOOST = {
    RegionMood.OFFENLAND: (Tag.FARM | Tag.NATURE | Tag.RURAL, 1.4),
    RegionMood.DORF: (Tag.FARM | Tag.RURAL | Tag.URBAN, 2.8),
    RegionMood.STADT: (Tag.URBAN | Tag.INDUSTRIAL | Tag.ROAD, 3.2),
    RegionMood.WALD: (Tag.FOREST | Tag.NATURE, 3.0),
}

MOOD_KIND_BOOST = {
    RegionMood.DORF: {
        FeatureKind.BAUERNHOF: 2.5, FeatureKind.HAUS: 4.5, FeatureKind.TIERE: 1.8,
        FeatureKind.KAPELLE: 2.5, FeatureKind.HEUBALLEN: 1.6, FeatureKind.ACKER: 1.2,
        FeatureKind.LATERNE: 2.5, FeatureKind.BUSCH: 1.2,
    },
    RegionMood.STADT: {
        FeatureKind.HAUS: 6.0, FeatureKind.INDUSTRIE: 4.0, FeatureKind.LAGERHALLE: 3.5,
        FeatureKind.LATERNE: 5.0, FeatureKind.HOCHSPANNUNG: 1.5, FeatureKind.WRACK: 1.2,
        FeatureKind.BAUM: 0.35, FeatureKind.ACKER: 0.15, FeatureKind.BUSCH: 0.4,
    },
    RegionMood.WALD: {
        FeatureKind.BAUM: 4.5, FeatureKind.BUSCH: 3.0, FeatureKind.FELSEN: 2.5,
        FeatureKind.SUMPF: 2.0, FeatureKind.RUINE: 1.5,
    },
    RegionMood.OFFENLAND: {
        FeatureKind.ACKER: 2.0, FeatureKind.HEUBALLEN: 1.5, FeatureKind.WINDRAD: 1.4,
        FeatureKind.TIERE: 1.3,
    },
}

FEATURE_DEPTH_WEIGHTS = {
    FeatureKind.WRACK: {Depth.NEAR: 1.0},
    FeatureKind.TIERE: {Depth.NEAR: 0.7, Depth.MID: 0.3},
    FeatureKind.HEUBALLEN: {Depth.NEAR: 0.6, Depth.MID: 0.4},
    FeatureKind.SUMPF: {Depth.NEAR: 1.0},
    FeatureKind.FELSEN: {Depth.NEAR: 0.3, Depth.MID: 0.5, Depth.FAR: 0.2},
    FeatureKind.ACKER: {Depth.NEAR: 1.0},
    FeatureKind.BAUERNHOF: {Depth.MID: 0.7, Depth.FAR: 0.3},
    FeatureKind.KAPELLE: {Depth.MID: 0.6, Depth.FAR: 0.4},
    FeatureKind.RUINE: {Depth.MID: 0.5, Depth.FAR: 0.5},
    FeatureKind.INDUSTRIE: {Depth.MID: 0.3, Depth.FAR: 0.7},
    FeatureKind.WINDRAD: {Depth.MID: 0.2, Depth.FAR: 0.8},
    FeatureKind.HOCHSPANNUNG: {Depth.MID: 0.3, Depth.FAR: 0.7},
    FeatureKind.HAUS: {Depth.NEAR: 0.4, Depth.MID: 0.6},
    FeatureKind.BAUM: {Depth.NEAR: 0.2, Depth.MID: 0.5, Depth.FAR: 0.3},
    FeatureKind.LATERNE: {Depth.NEAR: 1.0},
    FeatureKind.LAGERHALLE: {Depth.MID: 0.5, Depth.FAR: 0.5},
    FeatureKind.BUSCH: {Depth.NEAR: 0.5, Depth.MID: 0.5},
    FeatureKind.BRUECKE: {Depth.NEAR: 1.0},
    FeatureKind.FLUSS: {Depth.MID: 0.5, Depth.FAR: 0.5},
    FeatureKind.SEE: {Depth.MID: 0.35, Depth.FAR: 0.65},
}

FEATURE_WIDTH = {
    FeatureKind.ACKER: (500, 900), FeatureKind.TIERE: (280, 450),
    FeatureKind.FELSEN: (200, 350), FeatureKind.INDUSTRIE: (400, 700),
    FeatureKind.RUINE: (280, 480), FeatureKind.WRACK: (180, 300),
    FeatureKind.BAUERNHOF: (400, 650), FeatureKind.SUMPF: (450, 750),
    FeatureKind.WINDRAD: (200, 320), FeatureKind.HOCHSPANNUNG: (600, 1000),
    FeatureKind.KAPELLE: (220, 360), FeatureKind.HEUBALLEN: (220, 380),
    FeatureKind.HAUS: (180, 320), FeatureKind.BAUM: (120, 220),
    FeatureKind.LATERNE: (60, 100), FeatureKind.LAGERHALLE: (350, 550),
    FeatureKind.BUSCH: (100, 180),
    FeatureKind.BRUECKE: (200, 220),
    FeatureKind.FLUSS: (360, 400),
    FeatureKind.SEE: (360, 400),
}

MOOD_GAP = {
    RegionMood.OFFENLAND: (700, 2200),
    RegionMood.DORF: (120, 360),
    RegionMood.STADT: (70, 220),
    RegionMood.WALD: (150, 450),
}

MIN_PLACE_STRETCH = 4500.0
MIN_PLACE_GAP = 3000.0


@dataclass(frozen=True, slots=True)
class Feature:
    kind: FeatureKind
    start: float
    width: float
    depth: Depth = Depth.NEAR
    depth_t: float = 0.0
    tags: Tag = Tag.NATURE
    props: FeatureProps = field(default_factory=FeatureProps)

    @property
    def end(self):
        return self.start + self.width

    @property
    def parallax(self):
        return depth_from_t(self.depth_t)[0]

    @property
    def scale(self):
        return depth_from_t(self.depth_t)[1] * self.props.scale_mul

    @property
    def y_offset(self):
        return depth_from_t(self.depth_t)[2] + self.props.y_jitter


@dataclass(frozen=True, slots=True)
class Region:
    mood: RegionMood
    start: float
    width: float
    sign_text: str = ""

    @property
    def end(self):
        return self.start + self.width


def _weights_for_mood(mood, *, building_d=0.0, forest_d=0.0, farm_d=0.0, industrial_d=0.0):
    boost_tags, tag_mul = MOOD_TAG_BOOST[mood]
    kind_boost = MOOD_KIND_BOOST.get(mood, {})
    bd = max(0.0, min(1.0, building_d))
    fd = max(0.0, min(1.0, forest_d))
    farm = max(0.0, min(1.0, farm_d))
    ind = max(0.0, min(1.0, industrial_d))
    out = {}
    for kind, base in BASE_WEIGHTS.items():
        tags = FEATURE_TAGS[kind]
        w = base
        if tags & boost_tags:
            w *= tag_mul
        w *= kind_boost.get(kind, 1.0)
        if Tag.URBAN in tags or kind in (FeatureKind.HAUS, FeatureKind.LATERNE, FeatureKind.LAGERHALLE):
            w *= 0.35 + 1.9 * bd
        if Tag.INDUSTRIAL in tags or kind in (FeatureKind.INDUSTRIE, FeatureKind.HOCHSPANNUNG, FeatureKind.WINDRAD):
            w *= 0.4 + 2.0 * ind
        if Tag.FOREST in tags or kind in (FeatureKind.BAUM, FeatureKind.BUSCH, FeatureKind.SUMPF):
            w *= 0.35 + 1.8 * fd
        if Tag.FARM in tags or kind in (FeatureKind.ACKER, FeatureKind.HEUBALLEN, FeatureKind.TIERE, FeatureKind.BAUERNHOF):
            w *= 0.4 + 1.7 * farm
        out[kind] = max(w, 0.01) if base > 0 else 0.0
    return out


def _pick_kind(rng, weights):
    kinds = [k for k, w in weights.items() if w > 0] or list(weights.keys())
    w = [max(weights[k], 0.01) for k in kinds]
    return rng.choices(kinds, weights=w, k=1)[0]


def _pick_depth_t(rng, kind):
    weights = FEATURE_DEPTH_WEIGHTS.get(kind, {Depth.NEAR: 1.0})
    depths = list(weights.keys())
    depth = rng.choices(depths, weights=[weights[d] for d in depths], k=1)[0]
    lo, hi = DEPTH_T_RANGE[depth]
    return depth, rng.uniform(lo, hi)


def _pick_mood(rng):
    return rng.choices(
        [RegionMood.OFFENLAND, RegionMood.DORF, RegionMood.STADT, RegionMood.WALD],
        weights=[4.5, 0.9, 0.5, 1.0], k=1,
    )[0]


def mood_from_name(name):
    try:
        return RegionMood(name)
    except ValueError:
        return RegionMood.OFFENLAND


def _make_region(rng, x, mood=None):
    mood = mood or _pick_mood(rng)
    widths = {
        RegionMood.OFFENLAND: (5000, 12000), RegionMood.WALD: (3500, 7000),
        RegionMood.DORF: (5000, 9000), RegionMood.STADT: (6000, 11000),
    }
    lo, hi = widths[mood]
    return Region(mood=mood, start=x, width=rng.uniform(lo, hi), sign_text="")


@dataclass
class LandscapeRoute:
    features: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    assets_root: Path = field(default_factory=lambda: Path("assets/backgrounds/zones"))
    features_root: Path = field(default_factory=lambda: Path("assets/backgrounds/features"))
    seed: int | None = None
    mood_provider: Callable | None = None
    place_name_provider: Callable | None = None
    _region_end: float = 0.0
    _feature_x: float = 200.0
    _rng: random.Random = field(default_factory=random.Random)
    _generated_until: float = 0.0
    _building_density: float = 0.0
    _forest_density: float = 0.0
    _farm_density: float = 0.0
    _industrial_density: float = 0.0
    _building_height_scale: float = 1.0
    _skyline_scale: float = 1.0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def base_layers(self, scroll_speed, facing=1.0, mood=None):
        root = self.assets_root
        mood = mood or RegionMood.OFFENLAND
        mid_dir = {
            RegionMood.OFFENLAND: "felder", RegionMood.WALD: "wald",
            RegionMood.DORF: "dorf", RegionMood.STADT: "stadt",
        }.get(mood, "felder")
        layers = [
            BackgroundLayer(
                path=root / mid_dir / "mid.svg", z_index=1, parallax=0.32,
                scroll_x=-facing * scroll_speed * 0.32, repeat_x=True,
            ),
        ]
        objects = root / mid_dir / "objects.svg"
        if mood in (RegionMood.DORF, RegionMood.STADT) and objects.is_file():
            layers.append(BackgroundLayer(
                path=objects, z_index=2, parallax=0.55,
                scroll_x=-facing * scroll_speed * 0.55, repeat_x=True,
            ))
        layers.append(BackgroundLayer(
            path=root / "landstrasse" / "ground.svg", z_index=10, parallax=1.0,
            scroll_x=-facing * scroll_speed * 1.0, repeat_x=True,
        ))
        return tuple(layers)

    def feature_object_path(self, kind, props=None):
        if kind == FeatureKind.BRUECKE and props is not None:
            label = getattr(props, "label", "") or ""
            if label in ("start", "mid", "end"):
                p = self.features_root / f"bruecke_{label}.svg"
                if p.is_file():
                    return p
        return self.features_root / f"{kind.value}.svg"

    def _mood_at(self, wx):
        if self.mood_provider is not None:
            m = self.mood_provider(wx)
            if m is not None:
                return m
        for r in self.regions:
            if r.start <= wx < r.end:
                return r.mood
        return RegionMood.OFFENLAND

    def _extend_regions(self, until):
        while self._region_end < until:
            forced = self.mood_provider(self._region_end) if self.mood_provider else None
            if self.regions and forced is not None:
                prev = self.regions[-1]
                if prev.mood in (RegionMood.DORF, RegionMood.STADT) and forced != prev.mood:
                    if (self._region_end - prev.start) < MIN_PLACE_STRETCH:
                        forced = prev.mood
            reg = _make_region(self._rng, self._region_end, mood=forced)
            if reg.mood in (RegionMood.DORF, RegionMood.STADT) and self.place_name_provider:
                real = self.place_name_provider(reg.start)
                if real:
                    reg = Region(mood=reg.mood, start=reg.start, width=reg.width, sign_text=real)
            self.regions.append(reg)
            self._region_end = reg.end

    def _extend_features(self, until):
        while self._feature_x < until:
            mood = self._mood_at(self._feature_x)
            weights = _weights_for_mood(
                mood,
                building_d=float(getattr(self, "_building_density", 0) or 0),
                forest_d=float(getattr(self, "_forest_density", 0) or 0),
                farm_d=float(getattr(self, "_farm_density", 0) or 0),
                industrial_d=float(getattr(self, "_industrial_density", 0) or 0),
            )
            kind = _pick_kind(self._rng, weights)
            depth, depth_t = _pick_depth_t(self._rng, kind)
            if kind in (FeatureKind.ACKER, FeatureKind.SUMPF):
                depth, depth_t = Depth.NEAR, self._rng.uniform(0.0, 0.12)
            props = sample_props(self._rng, kind.value)
            if kind in (FeatureKind.HAUS, FeatureKind.BAUERNHOF, FeatureKind.LAGERHALLE,
                        FeatureKind.INDUSTRIE, FeatureKind.KAPELLE, FeatureKind.RUINE):
                hs = float(getattr(self, "_building_height_scale", 1.0) or 1.0)
                hs *= self._rng.uniform(0.88, 1.12)
                props = FeatureProps(
                    scale_mul=props.scale_mul * hs, flip_x=props.flip_x,
                    hue_shift=props.hue_shift, sat_mul=props.sat_mul,
                    bright_mul=props.bright_mul, alpha_mul=1.0,
                    y_jitter=props.y_jitter, style=props.style,
                    extra_a=props.extra_a, extra_b=props.extra_b, label=props.label,
                )
            lo, hi = FEATURE_WIDTH[kind]
            depth_scale = depth_from_t(depth_t)[1]
            fw = self._rng.uniform(lo, hi) / depth_scale * props.scale_mul
            self.features.append(Feature(
                kind=kind, start=self._feature_x, width=fw,
                depth=depth, depth_t=depth_t, tags=FEATURE_TAGS[kind], props=props,
            ))
            gap_lo, gap_hi = MOOD_GAP[mood]
            bd = float(getattr(self, "_building_density", 0) or 0)
            gap_mul = max(0.25, min(1.8, (1.0 - 0.60 * bd)))
            self._feature_x += fw * depth_scale + self._rng.uniform(gap_lo * gap_mul, gap_hi * gap_mul)

    def inject_water_hits(self, hits):
        """Continuous water bands + tiled bridges (start/mid/end)."""
        if not hits:
            return
        scale = float(getattr(self, "_geo_world_scale", 1.0) or 1.0)
        buckets = {}
        for hit in hits:
            side = getattr(hit, "side", "") or ""
            kind_s = getattr(hit, "kind", "water") or "water"
            if side == "right":
                continue
            key = (side, kind_s if kind_s in ("lake", "water") else "river")
            buckets.setdefault(key, []).append(hit)
        spans = []
        for (side, kind_s), group in buckets.items():
            pts = sorted(
                (float(getattr(h, "distance_m", 0) or 0),
                 float(getattr(h, "width_hint_m", 30) or 30))
                for h in group
            )
            if not pts:
                continue
            intervals = [[dm - max(40, w * 0.6), dm + max(40, w * 0.6)] for dm, w in pts]
            intervals.sort()
            merged = [intervals[0]]
            for a, b in intervals[1:]:
                if a <= merged[-1][1] + 250.0:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            for a, b in merged:
                spans.append((side, kind_s, a, b))
        existing = {(round(f.start, 0), f.kind) for f in self.features}
        for side, kind_s, start_m, end_m in spans:
            wx0, wx1 = start_m * scale, end_m * scale
            span_w = max(200.0, wx1 - wx0)
            if side == "cross":
                tile_w = 200.0
                n_mid = max(1, int(span_w / tile_w) - 1)
                step = span_w / max(1, (2 + n_mid) - 0.15)
                labels = ["start"] + ["mid"] * n_mid + ["end"]
                x = wx0
                for lab in labels:
                    if any(f.kind == FeatureKind.BRUECKE and abs(f.start - x) < tile_w * 0.4 for f in self.features):
                        x += step
                        continue
                    base = sample_props(self._rng, "bruecke")
                    props = FeatureProps(
                        scale_mul=base.scale_mul, flip_x=False,
                        hue_shift=base.hue_shift, sat_mul=base.sat_mul,
                        bright_mul=base.bright_mul, alpha_mul=1.0, y_jitter=0.0,
                        style=base.style, label=lab,
                    )
                    self.features.append(Feature(
                        kind=FeatureKind.BRUECKE, start=x, width=tile_w,
                        depth=Depth.NEAR, depth_t=0.04,
                        tags=FEATURE_TAGS[FeatureKind.BRUECKE], props=props,
                    ))
                    existing.add((round(x, 0), FeatureKind.BRUECKE))
                    x += step
                continue
            fk = FeatureKind.SEE if kind_s in ("lake", "water") else FeatureKind.FLUSS
            depth, depth_t = (Depth.FAR, 0.72) if fk == FeatureKind.SEE else (Depth.MID, 0.50)
            tile_w = 380.0
            n_tiles = max(2, int(span_w / tile_w) + 1)
            x = wx0
            for _ in range(n_tiles):
                if not any(f.kind == fk and abs(f.start - x) < tile_w * 0.5 for f in self.features):
                    self.features.append(Feature(
                        kind=fk, start=x, width=tile_w, depth=depth, depth_t=depth_t,
                        tags=FEATURE_TAGS[fk], props=sample_props(self._rng, fk.value),
                    ))
                x += tile_w * 0.92
        self.features.sort(key=lambda f: f.start)

    def ensure_ahead(self, distance, look_ahead=12000.0):
        target = distance + look_ahead
        if target <= self._generated_until:
            return
        self._extend_regions(target + 5000)
        self._extend_features(target)
        self._generated_until = target

    def prune_behind(self, distance, keep_behind=8000.0):
        cutoff = distance - keep_behind
        if cutoff <= 0:
            return
        self.features = [f for f in self.features if f.end >= cutoff]
        self.regions = [r for r in self.regions if r.end >= cutoff]

    def active_features(self, distance, margin=1200.0):
        self.ensure_ahead(distance)
        out = [f for f in self.features
               if (f.start - margin / max(f.parallax, 0.15)) <= distance <= (f.end + margin / max(f.parallax, 0.15))]
        out.sort(key=lambda f: (f.parallax, f.y_offset, f.start))
        return out

    def active_regions(self, distance, margin=800.0):
        self.ensure_ahead(distance)
        return [r for r in self.regions if r.mood != RegionMood.OFFENLAND
                and (r.start - margin) <= distance <= (r.end + margin)]

    @property
    def overlays(self):
        return self.regions


def generate_route(length=15000.0, seed=None):
    route = LandscapeRoute(seed=seed)
    route.ensure_ahead(0.0, look_ahead=max(length, 15000.0))
    return route


def default_german_route(seed=42):
    return generate_route(length=15000.0, seed=seed)
