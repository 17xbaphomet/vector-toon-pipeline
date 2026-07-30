"""Continuous base landscape + place overlays + random individual features."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from .value_objects import BackgroundLayer


class ZoneId(str, Enum):
    FELDER = "felder"
    LANDSTRASSE = "landstrasse"
    WALD = "wald"
    DORF = "dorf"
    STADT = "stadt"


class FeatureKind(str, Enum):
    """Sparse individual landscape props."""

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


OVERLAY_TYPES: frozenset[ZoneId] = frozenset({ZoneId.DORF, ZoneId.STADT, ZoneId.WALD})

SIGN_NAMES: dict[ZoneId, list[str]] = {
    ZoneId.DORF: ["Musterdorf", "Kleinhausen", "Bergheim", "Lindenau", "Schönbach"],
    ZoneId.STADT: ["Neustadt", "Altenburg", "Rheinfeld", "Hochstadt", "Mühlheim"],
    ZoneId.WALD: ["Stadtwald", "Eichenforst", "Tannengrund", "Birkenhain"],
}

# Relative spawn weights (higher = more common)
FEATURE_WEIGHTS: dict[FeatureKind, float] = {
    FeatureKind.ACKER: 1.4,
    FeatureKind.TIERE: 1.2,
    FeatureKind.FELSEN: 1.0,
    FeatureKind.BAUERNHOF: 0.9,
    FeatureKind.HEUBALLEN: 1.1,
    FeatureKind.WINDRAD: 0.7,
    FeatureKind.HOCHSPANNUNG: 0.6,
    FeatureKind.SUMPF: 0.5,
    FeatureKind.RUINE: 0.45,
    FeatureKind.WRACK: 0.4,
    FeatureKind.INDUSTRIE: 0.35,
    FeatureKind.KAPELLE: 0.4,
}

FEATURE_WIDTH: dict[FeatureKind, tuple[float, float]] = {
    FeatureKind.ACKER: (500, 900),
    FeatureKind.TIERE: (280, 450),
    FeatureKind.FELSEN: (200, 350),
    FeatureKind.INDUSTRIE: (400, 700),
    FeatureKind.RUINE: (280, 480),
    FeatureKind.WRACK: (180, 300),
    FeatureKind.BAUERNHOF: (400, 650),
    FeatureKind.SUMPF: (450, 750),
    FeatureKind.WINDRAD: (200, 320),
    FeatureKind.HOCHSPANNUNG: (600, 1000),
    FeatureKind.KAPELLE: (220, 360),
    FeatureKind.HEUBALLEN: (220, 380),
}


@dataclass(frozen=True, slots=True)
class Overlay:
    kind: ZoneId
    start: float
    width: float
    sign_text: str

    @property
    def end(self) -> float:
        return self.start + self.width

    @property
    def sign_world_x(self) -> float:
        return self.start - 40.0


@dataclass(frozen=True, slots=True)
class Feature:
    """A single sparse landscape prop planted at world-x."""

    kind: FeatureKind
    start: float
    width: float

    @property
    def end(self) -> float:
        return self.start + self.width


@dataclass
class LandscapeRoute:
    overlays: Sequence[Overlay]
    features: Sequence[Feature] = ()
    assets_root: Path = field(default_factory=lambda: Path("assets/backgrounds/zones"))
    features_root: Path = field(default_factory=lambda: Path("assets/backgrounds/features"))
    seed: int | None = None

    def base_layers(self, scroll_speed: float, facing: float = 1.0) -> tuple[BackgroundLayer, ...]:
        root = self.assets_root
        return (
            BackgroundLayer(
                path=root / "felder" / "mid.svg",
                z_index=1,
                parallax=0.35,
                scroll_x=-facing * scroll_speed * 0.35,
                repeat_x=True,
            ),
            BackgroundLayer(
                path=root / "landstrasse" / "ground.svg",
                z_index=2,
                parallax=1.0,
                scroll_x=-facing * scroll_speed * 1.0,
                repeat_x=True,
            ),
        )

    def overlay_object_path(self, kind: ZoneId) -> Path:
        return self.assets_root / kind.value / "objects.svg"

    def feature_object_path(self, kind: FeatureKind) -> Path:
        return self.features_root / f"{kind.value}.svg"

    def active_overlays(self, distance: float, margin: float = 2000.0) -> list[Overlay]:
        return [
            o
            for o in self.overlays
            if (o.start - margin) <= distance <= (o.end + margin)
        ]

    def active_features(self, distance: float, margin: float = 1200.0) -> list[Feature]:
        return [
            f
            for f in self.features
            if (f.start - margin) <= distance <= (f.end + margin)
        ]


def _pick_feature(rng: random.Random) -> FeatureKind:
    kinds = list(FEATURE_WEIGHTS.keys())
    weights = [FEATURE_WEIGHTS[k] for k in kinds]
    return rng.choices(kinds, weights=weights, k=1)[0]


def generate_route(
    length: float = 80000.0,
    seed: int | None = None,
    min_gap: float = 8000.0,
    max_gap: float = 15000.0,
    min_width: float = 2500.0,
    max_width: float = 4500.0,
    feature_min_gap: float = 600.0,
    feature_max_gap: float = 2200.0,
) -> LandscapeRoute:
    """Large place overlays + denser random individual features."""
    rng = random.Random(seed)

    # ── large places ─────────────────────────────────────────────────
    overlays: list[Overlay] = []
    x = rng.uniform(5000, 9000)
    place_kinds = [ZoneId.DORF, ZoneId.STADT, ZoneId.WALD]
    while x < length:
        kind = rng.choice(place_kinds)
        width = rng.uniform(min_width, max_width)
        names = SIGN_NAMES.get(kind, [kind.value.title()])
        text = rng.choice(names)
        overlays.append(Overlay(kind=kind, start=x, width=width, sign_text=text))
        x += width + rng.uniform(min_gap, max_gap)

    # ── sparse individual features (avoid heavy overlap with places) ─
    place_ranges = [(o.start - 200, o.end + 200) for o in overlays]

    def _inside_place(wx: float) -> bool:
        return any(a <= wx <= b for a, b in place_ranges)

    features: list[Feature] = []
    fx = rng.uniform(400, 1200)
    while fx < length:
        if not _inside_place(fx):
            kind = _pick_feature(rng)
            lo, hi = FEATURE_WIDTH[kind]
            fw = rng.uniform(lo, hi)
            features.append(Feature(kind=kind, start=fx, width=fw))
            fx += fw + rng.uniform(feature_min_gap, feature_max_gap)
        else:
            fx += rng.uniform(300, 800)

    return LandscapeRoute(
        overlays=tuple(overlays),
        features=tuple(features),
        seed=seed,
    )


def default_german_tour(seed: int | None = 42) -> LandscapeRoute:
    return generate_route(length=80000.0, seed=seed)
