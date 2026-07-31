"""Organic infinite landscape: tagged features + combinable props, forever.

Optional geo mood provider biases soft regions from real density climate
(no exact object placement).

Ortsschilder use only real place names from the map (place_name_provider);
never synthetic placeholders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from typing import Callable, Sequence

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


class Depth(int, Enum):
    NEAR = 0
    MID = 1
    FAR = 2


class RegionMood(str, Enum):
    OFFENLAND = "offenland"
    DORF = "dorf"
    STADT = "stadt"
    WALD = "wald"


DEPTH_PARAMS: dict[Depth, tuple[float, float, float]] = {
    Depth.NEAR: (1.00, 1.00, 0.0),
    Depth.MID: (0.55, 0.70, -35.0),
    Depth.FAR: (0.28, 0.45, -70.0),
}

FEATURE_TAGS: dict[FeatureKind, Tag] = {
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
}

BASE_WEIGHTS: dict[FeatureKind, float] = {
    FeatureKind.ACKER: 1.3, FeatureKind.TIERE: 0.9, FeatureKind.FELSEN: 0.7,
    FeatureKind.INDUSTRIE: 0.15, FeatureKind.RUINE: 0.25, FeatureKind.WRACK: 0.2,
    FeatureKind.BAUERNHOF: 0.5, FeatureKind.SUMPF: 0.3, FeatureKind.WINDRAD: 0.35,
    FeatureKind.HOCHSPANNUNG: 0.3, FeatureKind.KAPELLE: 0.25, FeatureKind.HEUBALLEN: 1.0,
    FeatureKind.HAUS: 0.2, FeatureKind.BAUM: 0.8, FeatureKind.LATERNE: 0.1,
    FeatureKind.LAGERHALLE: 0.1, FeatureKind.BUSCH: 0.9,
}

MOOD_TAG_BOOST: dict[RegionMood, tuple[Tag, float]] = {
    RegionMood.OFFENLAND: (Tag.FARM | Tag.NATURE | Tag.RURAL, 1.4),
    RegionMood.DORF: (Tag.FARM | Tag.RURAL | Tag.URBAN, 2.8),
    RegionMood.STADT: (Tag.URBAN | Tag.INDUSTRIAL | Tag.ROAD, 3.2),
    RegionMood.WALD: (Tag.FOREST | Tag.NATURE, 3.0),
}

MOOD_KIND_BOOST: dict[RegionMood, dict[FeatureKind, float]] = {
    RegionMood.DORF: {
        FeatureKind.BAUERNHOF: 2.5, FeatureKind.HAUS: 3.0, FeatureKind.TIERE: 2.0,
        FeatureKind.KAPELLE: 2.5, FeatureKind.HEUBALLEN: 1.8, FeatureKind.ACKER: 1.5,
        FeatureKind.LATERNE: 1.5,
    },
    RegionMood.STADT: {
        FeatureKind.HAUS: 4.0, FeatureKind.INDUSTRIE: 3.5, FeatureKind.LAGERHALLE: 3.0,
        FeatureKind.LATERNE: 3.5, FeatureKind.HOCHSPANNUNG: 2.0, FeatureKind.WRACK: 1.5,
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

FEATURE_DEPTH_WEIGHTS: dict[FeatureKind, dict[Depth, float]] = {
    FeatureKind.WRACK: {Depth.NEAR: 1.0},
    FeatureKind.TIERE: {Depth.NEAR: 0.7, Depth.MID: 0.3},
    FeatureKind.HEUBALLEN: {Depth.NEAR: 0.6, Depth.MID: 0.4},
    FeatureKind.SUMPF: {Depth.NEAR: 0.4, Depth.MID: 0.6},
    FeatureKind.FELSEN: {Depth.NEAR: 0.3, Depth.MID: 0.5, Depth.FAR: 0.2},
    FeatureKind.ACKER: {Depth.MID: 0.6, Depth.FAR: 0.4},
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
}

FEATURE_WIDTH: dict[FeatureKind, tuple[float, float]] = {
    FeatureKind.ACKER: (500, 900), FeatureKind.TIERE: (280, 450),
    FeatureKind.FELSEN: (200, 350), FeatureKind.INDUSTRIE: (400, 700),
    FeatureKind.RUINE: (280, 480), FeatureKind.WRACK: (180, 300),
    FeatureKind.BAUERNHOF: (400, 650), FeatureKind.SUMPF: (450, 750),
    FeatureKind.WINDRAD: (200, 320), FeatureKind.HOCHSPANNUNG: (600, 1000),
    FeatureKind.KAPELLE: (220, 360), FeatureKind.HEUBALLEN: (220, 380),
    FeatureKind.HAUS: (180, 320), FeatureKind.BAUM: (120, 220),
    FeatureKind.LATERNE: (60, 100), FeatureKind.LAGERHALLE: (350, 550),
    FeatureKind.BUSCH: (100, 180),
}

MOOD_GAP: dict[RegionMood, tuple[float, float]] = {
    RegionMood.OFFENLAND: (700, 2200),
    RegionMood.DORF: (180, 520),
    RegionMood.STADT: (120, 380),
    RegionMood.WALD: (150, 450),
}


@dataclass(frozen=True, slots=True)
class Feature:
    kind: FeatureKind
    start: float
    width: float
    depth: Depth = Depth.NEAR
    tags: Tag = Tag.NATURE
    props: FeatureProps = field(default_factory=FeatureProps)

    @property
    def end(self) -> float:
        return self.start + self.width

    @property
    def parallax(self) -> float:
        return DEPTH_PARAMS[self.depth][0]

    @property
    def scale(self) -> float:
        return DEPTH_PARAMS[self.depth][1] * self.props.scale_mul

    @property
    def y_offset(self) -> float:
        return DEPTH_PARAMS[self.depth][2] + self.props.y_jitter


@dataclass(frozen=True, slots=True)
class Region:
    """Settlement / forest stretch with German-style entrance + exit signs.

    sign_text is only set when a real map place name is known.
    """

    mood: RegionMood
    start: float
    width: float
    sign_text: str = ""

    @property
    def end(self) -> float:
        return self.start + self.width

    @property
    def sign_world_x(self) -> float:
        """Ortseingang (Zeichen 310) — just before the settlement."""
        return self.start - 40.0

    @property
    def entrance_world_x(self) -> float:
        return self.sign_world_x

    @property
    def exit_world_x(self) -> float:
        """Ortsausgang (Zeichen 311) — just after the settlement."""
        return self.end + 20.0


def _weights_for_mood(mood: RegionMood) -> dict[FeatureKind, float]:
    boost_tags, tag_mul = MOOD_TAG_BOOST[mood]
    kind_boost = MOOD_KIND_BOOST.get(mood, {})
    out: dict[FeatureKind, float] = {}
    for kind, base in BASE_WEIGHTS.items():
        tags = FEATURE_TAGS[kind]
        w = base
        if tags & boost_tags:
            w *= tag_mul
        w *= kind_boost.get(kind, 1.0)
        if mood == RegionMood.WALD and Tag.URBAN in tags and Tag.FOREST not in tags:
            w *= 0.15
        if mood == RegionMood.STADT and Tag.FOREST in tags and Tag.URBAN not in tags:
            w *= 0.2
        if mood == RegionMood.DORF and Tag.INDUSTRIAL in tags and Tag.FARM not in tags:
            w *= 0.35
        out[kind] = max(w, 0.01)
    return out


def _pick_kind(rng: random.Random, weights: dict[FeatureKind, float]) -> FeatureKind:
    kinds = list(weights.keys())
    w = [weights[k] for k in kinds]
    return rng.choices(kinds, weights=w, k=1)[0]


def _pick_depth(rng: random.Random, kind: FeatureKind) -> Depth:
    weights = FEATURE_DEPTH_WEIGHTS.get(kind, {Depth.NEAR: 1.0})
    depths = list(weights.keys())
    w = [weights[d] for d in depths]
    return rng.choices(depths, weights=w, k=1)[0]


def _pick_mood(rng: random.Random) -> RegionMood:
    return rng.choices(
        [RegionMood.OFFENLAND, RegionMood.DORF, RegionMood.STADT, RegionMood.WALD],
        weights=[3.0, 1.4, 0.9, 1.2], k=1,
    )[0]


def mood_from_name(name: str) -> RegionMood:
    try:
        return RegionMood(name)
    except ValueError:
        return RegionMood.OFFENLAND


def _make_region(rng: random.Random, x: float, mood: RegionMood | None = None) -> Region:
    """Build a region. sign_text stays empty until a real map name is resolved."""
    mood = mood or _pick_mood(rng)
    if mood == RegionMood.OFFENLAND:
        width = rng.uniform(4000, 9000)
    elif mood == RegionMood.WALD:
        width = rng.uniform(2500, 5500)
    elif mood == RegionMood.DORF:
        width = rng.uniform(1800, 3500)
    else:
        width = rng.uniform(2200, 4500)
    return Region(mood=mood, start=x, width=width, sign_text="")


@dataclass
class LandscapeRoute:
    features: list[Feature] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    assets_root: Path = field(default_factory=lambda: Path("assets/backgrounds/zones"))
    features_root: Path = field(default_factory=lambda: Path("assets/backgrounds/features"))
    seed: int | None = None
    mood_provider: Callable[[float], RegionMood | None] | None = None
    # Optional: world_x → real place name (from map / Nominatim)
    place_name_provider: Callable[[float], str | None] | None = None
    _region_end: float = 0.0
    _feature_x: float = 200.0
    _rng: random.Random = field(default_factory=random.Random)
    _generated_until: float = 0.0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def base_layers(self, scroll_speed: float, facing: float = 1.0) -> tuple[BackgroundLayer, ...]:
        root = self.assets_root
        return (
            BackgroundLayer(
                path=root / "felder" / "mid.svg", z_index=1, parallax=0.35,
                scroll_x=-facing * scroll_speed * 0.35, repeat_x=True,
            ),
            BackgroundLayer(
                path=root / "landstrasse" / "ground.svg", z_index=2, parallax=1.0,
                scroll_x=-facing * scroll_speed * 1.0, repeat_x=True,
            ),
        )

    def feature_object_path(self, kind: FeatureKind) -> Path:
        return self.features_root / f"{kind.value}.svg"

    def _mood_at(self, wx: float) -> RegionMood:
        if self.mood_provider is not None:
            m = self.mood_provider(wx)
            if m is not None:
                return m
        for r in self.regions:
            if r.start <= wx < r.end:
                return r.mood
        return RegionMood.OFFENLAND

    def _extend_regions(self, until: float) -> None:
        while self._region_end < until:
            forced = None
            if self.mood_provider is not None:
                forced = self.mood_provider(self._region_end)
            reg = _make_region(self._rng, self._region_end, mood=forced)
            # Only real map names — never synthetic placeholders
            if (
                reg.mood != RegionMood.OFFENLAND
                and self.place_name_provider is not None
            ):
                real = self.place_name_provider(reg.start)
                if real:
                    reg = Region(
                        mood=reg.mood, start=reg.start, width=reg.width, sign_text=real
                    )
            self.regions.append(reg)
            self._region_end = reg.end

    def resolve_place_names(self) -> None:
        """Fill missing sign_text from the map provider (real names only)."""
        if self.place_name_provider is None:
            return
        updated: list[Region] = []
        changed = False
        for reg in self.regions:
            if reg.mood == RegionMood.OFFENLAND or reg.sign_text:
                updated.append(reg)
                continue
            real = self.place_name_provider(reg.start)
            if real:
                updated.append(
                    Region(mood=reg.mood, start=reg.start, width=reg.width, sign_text=real)
                )
                changed = True
            else:
                updated.append(reg)
        if changed:
            self.regions = updated

    def _extend_features(self, until: float) -> None:
        while self._feature_x < until:
            mood = self._mood_at(self._feature_x)
            weights = _weights_for_mood(mood)
            kind = _pick_kind(self._rng, weights)
            depth = _pick_depth(self._rng, kind)
            props = sample_props(self._rng, kind.value)
            lo, hi = FEATURE_WIDTH[kind]
            depth_scale = DEPTH_PARAMS[depth][1]
            fw = self._rng.uniform(lo, hi) / depth_scale * props.scale_mul
            self.features.append(
                Feature(
                    kind=kind, start=self._feature_x, width=fw,
                    depth=depth, tags=FEATURE_TAGS[kind], props=props,
                )
            )
            gap_lo, gap_hi = MOOD_GAP[mood]
            self._feature_x += fw * depth_scale + self._rng.uniform(gap_lo, gap_hi)

    def ensure_ahead(self, distance: float, look_ahead: float = 12000.0) -> None:
        target = distance + look_ahead
        if target <= self._generated_until:
            return
        self._extend_regions(target + 5000)
        self._extend_features(target)
        self._generated_until = target

    def prune_behind(self, distance: float, keep_behind: float = 8000.0) -> None:
        cutoff = distance - keep_behind
        if cutoff <= 0:
            return
        self.features = [f for f in self.features if f.end >= cutoff]
        self.regions = [r for r in self.regions if r.end >= cutoff]

    def active_features(self, distance: float, margin: float = 1200.0) -> list[Feature]:
        self.ensure_ahead(distance)
        out: list[Feature] = []
        for f in self.features:
            m = margin / max(f.parallax, 0.15)
            if (f.start - m) <= distance <= (f.end + m):
                out.append(f)
        out.sort(key=lambda f: (f.parallax, f.start))
        return out

    def active_regions(self, distance: float, margin: float = 800.0) -> list[Region]:
        self.ensure_ahead(distance)
        return [
            r for r in self.regions
            if r.mood != RegionMood.OFFENLAND
            and (r.start - margin) <= distance <= (r.end + margin)
        ]

    @property
    def overlays(self) -> Sequence[Region]:
        return self.regions


def generate_route(length: float = 15000.0, seed: int | None = None) -> LandscapeRoute:
    route = LandscapeRoute(seed=seed)
    route.ensure_ahead(0.0, look_ahead=max(length, 15000.0))
    return route


def default_german_route(seed: int | None = 42) -> LandscapeRoute:
    return generate_route(length=15000.0, seed=seed)
