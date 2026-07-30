"""German landscapes: continuous base + walk-into overlays (no fade).

Cities, villages and forest patches OVERLAY the base countryside so the
character can walk in and out. Placement is generic and randomized with
large gaps. Ortsschild props are world-fixed on the ground plane.
"""

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


# Features that sit ON TOP of the continuous base landscape
OVERLAY_TYPES: frozenset[ZoneId] = frozenset({ZoneId.DORF, ZoneId.STADT, ZoneId.WALD})

# Default sign names (German places)
SIGN_NAMES: dict[ZoneId, list[str]] = {
    ZoneId.DORF: ["Musterdorf", "Kleinhausen", "Bergheim", "Lindenau", "Schönbach"],
    ZoneId.STADT: ["Neustadt", "Altenburg", "Rheinfeld", "Hochstadt", "Mühlheim"],
    ZoneId.WALD: ["Stadtwald", "Eichenforst", "Tannengrund", "Birkenhain"],
}


@dataclass(frozen=True, slots=True)
class Overlay:
    """A landscape feature planted at a fixed world-x range."""

    kind: ZoneId
    start: float          # world distance where overlay begins
    width: float          # length of the overlay in world units
    sign_text: str

    @property
    def end(self) -> float:
        return self.start + self.width

    @property
    def sign_world_x(self) -> float:
        """Ortsschild sits just before the entrance."""
        return self.start - 30.0


@dataclass
class LandscapeRoute:
    """
    Continuous base countryside + randomly placed overlays.

    No ordered sequence, no crossfade — overlays appear when the walker
    reaches their world-x and disappear when leaving.
    """

    overlays: Sequence[Overlay]
    assets_root: Path = field(default_factory=lambda: Path("assets/backgrounds/zones"))
    seed: int | None = None

    # ── base layers (always on) ──────────────────────────────────────

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
            # Country road as continuous ground
            BackgroundLayer(
                path=root / "landstrasse" / "ground.svg",
                z_index=2,
                parallax=1.0,
                scroll_x=-facing * scroll_speed * 1.0,
                repeat_x=True,
            ),
        )

    def overlay_layers(
        self, kind: ZoneId, scroll_speed: float, facing: float = 1.0
    ) -> tuple[BackgroundLayer, ...]:
        """Mid (+ optional ground accent) for an overlay type — non-tiling placement handled by caller."""
        root = self.assets_root / kind.value
        layers = [
            BackgroundLayer(
                path=root / "mid.svg",
                z_index=5,
                parallax=1.0,  # locked to ground so it doesn't float
                scroll_x=0.0,  # position set explicitly via world offset
                repeat_x=False,
            ),
        ]
        # Wald also darkens the ground strip a bit
        if kind == ZoneId.WALD:
            layers.append(
                BackgroundLayer(
                    path=root / "ground.svg",
                    z_index=4,
                    parallax=1.0,
                    scroll_x=0.0,
                    repeat_x=False,
                )
            )
        return tuple(layers)

    def active_overlays(self, distance: float, margin: float = 900.0) -> list[Overlay]:
        """Overlays whose world range is near the camera (visible or about to be)."""
        return [
            o
            for o in self.overlays
            if (o.start - margin) <= distance <= (o.end + margin)
        ]

    def near_sign(self, distance: float, window: float = 200.0) -> Overlay | None:
        """Return overlay whose sign is within window of current distance."""
        best: Overlay | None = None
        best_d = window
        for o in self.overlays:
            d = abs(distance - o.sign_world_x)
            if d < best_d:
                best_d = d
                best = o
        return best


def generate_route(
    length: float = 20000.0,
    seed: int | None = None,
    min_gap: float = 1200.0,
    max_gap: float = 2800.0,
    min_width: float = 700.0,
    max_width: float = 1400.0,
) -> LandscapeRoute:
    """
    Place overlays randomly along the road with large gaps.

    No fixed order — each pick is independent from {dorf, stadt, wald}.
    """
    rng = random.Random(seed)
    overlays: list[Overlay] = []
    x = rng.uniform(600, 1000)  # first feature not right at start

    kinds = [ZoneId.DORF, ZoneId.STADT, ZoneId.WALD]
    while x < length:
        kind = rng.choice(kinds)
        width = rng.uniform(min_width, max_width)
        names = SIGN_NAMES.get(kind, [kind.value.title()])
        text = rng.choice(names)
        overlays.append(Overlay(kind=kind, start=x, width=width, sign_text=text))
        x += width + rng.uniform(min_gap, max_gap)

    return LandscapeRoute(overlays=tuple(overlays), seed=seed)


# Back-compat alias used by older imports
def default_german_tour(seed: int | None = 42) -> LandscapeRoute:
    return generate_route(length=25000.0, seed=seed)
