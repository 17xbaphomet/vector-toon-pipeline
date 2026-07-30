"""German landscape zones and generic transition system."""

from __future__ import annotations

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


INNERORTS: frozenset[ZoneId] = frozenset({ZoneId.DORF, ZoneId.STADT})


@dataclass(frozen=True, slots=True)
class ZoneDef:
    id: ZoneId
    label: str
    sky: str = "sky.svg"
    mid: str = "mid.svg"
    ground: str = "ground.svg"


ZONE_CATALOG: dict[ZoneId, ZoneDef] = {
    ZoneId.FELDER: ZoneDef(ZoneId.FELDER, "Felder"),
    ZoneId.LANDSTRASSE: ZoneDef(ZoneId.LANDSTRASSE, "Landstraße"),
    ZoneId.WALD: ZoneDef(ZoneId.WALD, "Wald"),
    ZoneId.DORF: ZoneDef(ZoneId.DORF, "Musterdorf"),
    ZoneId.STADT: ZoneDef(ZoneId.STADT, "Neustadt"),
}


@dataclass(frozen=True, slots=True)
class ZoneSegment:
    zone: ZoneId
    start: float
    end: float
    sign_text: str | None = None


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    at_distance: float
    from_zone: ZoneId
    to_zone: ZoneId
    show_sign: bool
    sign_text: str
    fade_s: float = 1.2


@dataclass
class ZoneSequence:
    segments: Sequence[ZoneSegment]
    assets_root: Path = field(default_factory=lambda: Path("assets/backgrounds/zones"))

    def zone_at(self, distance: float) -> ZoneSegment:
        if not self.segments:
            raise ValueError("ZoneSequence has no segments")
        for seg in self.segments:
            if seg.start <= distance < seg.end:
                return seg
        return self.segments[-1]

    def active_blend(
        self, distance: float, scroll_speed: float, fade_s: float = 1.2
    ) -> tuple[ZoneSegment, ZoneSegment | None, float]:
        cur = self.zone_at(distance)
        fade_dist = max(fade_s * scroll_speed, 1.0)
        nxt: ZoneSegment | None = None
        for seg in self.segments:
            if seg.start >= cur.end - 1e-6 and seg is not cur:
                nxt = seg
                break
        if nxt is None:
            return cur, None, 0.0
        remaining = cur.end - distance
        if remaining > fade_dist:
            return cur, nxt, 0.0
        blend = 1.0 - (remaining / fade_dist)
        return cur, nxt, max(0.0, min(1.0, blend))

    def transitions(self) -> list[TransitionEvent]:
        events: list[TransitionEvent] = []
        for a, b in zip(self.segments[:-1], self.segments[1:]):
            entering_inner = b.zone in INNERORTS and a.zone not in INNERORTS
            leaving_inner = a.zone in INNERORTS and b.zone not in INNERORTS
            if entering_inner:
                text = b.sign_text or ZONE_CATALOG[b.zone].label
            elif leaving_inner:
                text = a.sign_text or ZONE_CATALOG[a.zone].label
            else:
                text = b.sign_text or ZONE_CATALOG[b.zone].label
            events.append(
                TransitionEvent(
                    at_distance=b.start,
                    from_zone=a.zone,
                    to_zone=b.zone,
                    show_sign=True,
                    sign_text=text,
                )
            )
        return events

    def layers_for(
        self, zone: ZoneId, scroll_speed: float, facing: float = 1.0
    ) -> tuple[BackgroundLayer, ...]:
        zdef = ZONE_CATALOG[zone]
        base = self.assets_root / zone.value
        return (
            BackgroundLayer(
                path=base / zdef.sky,
                z_index=0,
                parallax=0.12,
                scroll_x=-facing * scroll_speed * 0.12,
                repeat_x=True,
            ),
            BackgroundLayer(
                path=base / zdef.mid,
                z_index=1,
                parallax=0.45,
                scroll_x=-facing * scroll_speed * 0.45,
                repeat_x=True,
            ),
            BackgroundLayer(
                path=base / zdef.ground,
                z_index=2,
                parallax=1.0,
                scroll_x=-facing * scroll_speed * 1.0,
                repeat_x=True,
            ),
        )


def default_german_tour() -> ZoneSequence:
    """Felder → Landstraße → Musterdorf → Stadtwald → Neustadt → Felder."""
    return ZoneSequence(
        segments=(
            ZoneSegment(ZoneId.FELDER, 0, 400),
            ZoneSegment(ZoneId.LANDSTRASSE, 400, 900, sign_text="Landstraße"),
            ZoneSegment(ZoneId.DORF, 900, 1600, sign_text="Musterdorf"),
            ZoneSegment(ZoneId.WALD, 1600, 2300, sign_text="Stadtwald"),
            ZoneSegment(ZoneId.STADT, 2300, 3200, sign_text="Neustadt"),
            ZoneSegment(ZoneId.FELDER, 3200, 99999, sign_text="Felder"),
        )
    )
