"""German landscapes: continuous base + walk-into overlays (no fade)."""

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


OVERLAY_TYPES: frozenset[ZoneId] = frozenset({ZoneId.DORF, ZoneId.STADT, ZoneId.WALD})

SIGN_NAMES: dict[ZoneId, list[str]] = {
    ZoneId.DORF: ["Musterdorf", "Kleinhausen", "Bergheim", "Lindenau", "Schönbach"],
    ZoneId.STADT: ["Neustadt", "Altenburg", "Rheinfeld", "Hochstadt", "Mühlheim"],
    ZoneId.WALD: ["Stadtwald", "Eichenforst", "Tannengrund", "Birkenhain"],
}

# Opaque fill behind overlay art so base never shows through
OVERLAY_FILL: dict[ZoneId, tuple[int, int, int, int]] = {
    ZoneId.DORF: (180, 210, 160, 255),
    ZoneId.STADT: (160, 175, 185, 255),
    ZoneId.WALD: (45, 80, 40, 255),
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


@dataclass
class LandscapeRoute:
    overlays: Sequence[Overlay]
    assets_root: Path = field(default_factory=lambda: Path("assets/backgrounds/zones"))
    seed: int | None = None

    def base_layers(self, scroll_speed: float, facing: float = 1.0) -> tuple[BackgroundLayer, ...]:
        root = self.assets_root
        return (
            BackgroundLayer(
                path=root / "felder" / "sky.svg",
                z_index=0,
                parallax=0.10,
                scroll_x=-facing * scroll_speed * 0.10,
                repeat_x=True,
            ),
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

    def overlay_asset_paths(self, kind: ZoneId) -> list[Path]:
        """Full stack so overlay fully covers base (sky + mid + ground)."""
        root = self.assets_root / kind.value
        return [root / "sky.svg", root / "mid.svg", root / "ground.svg"]

    def active_overlays(self, distance: float, margin: float = 2000.0) -> list[Overlay]:
        return [
            o
            for o in self.overlays
            if (o.start - margin) <= distance <= (o.end + margin)
        ]


def generate_route(
    length: float = 80000.0,
    seed: int | None = None,
    min_gap: float = 8000.0,
    max_gap: float = 15000.0,
    min_width: float = 2500.0,
    max_width: float = 4500.0,
) -> LandscapeRoute:
    """
    Sparse random overlays. At ~133 px/s:
      gap 8k–15k  → 60–110 s between places
      width 2.5k–4.5k → 20–35 s walking through
    """
    rng = random.Random(seed)
    overlays: list[Overlay] = []
    # First place after a long open stretch
    x = rng.uniform(5000, 9000)

    kinds = [ZoneId.DORF, ZoneId.STADT, ZoneId.WALD]
    while x < length:
        kind = rng.choice(kinds)
        width = rng.uniform(min_width, max_width)
        names = SIGN_NAMES.get(kind, [kind.value.title()])
        text = rng.choice(names)
        overlays.append(Overlay(kind=kind, start=x, width=width, sign_text=text))
        x += width + rng.uniform(min_gap, max_gap)

    return LandscapeRoute(overlays=tuple(overlays), seed=seed)


def default_german_tour(seed: int | None = 42) -> LandscapeRoute:
    return generate_route(length=80000.0, seed=seed)
