from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math
from typing import Mapping, Sequence


class Viseme(str, Enum):
    """Preston Blair mouth shapes used by Rhubarb."""

    X = "X"  # rest / closed
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"


@dataclass(frozen=True, slots=True)
class Affine:
    """SVG-compatible 2D affine transform (matrix a b c d e f)."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0  # tx
    f: float = 0.0  # ty

    def to_svg_matrix(self) -> str:
        return f"matrix({self.a:.6g} {self.b:.6g} {self.c:.6g} {self.d:.6g} {self.e:.6g} {self.f:.6g})"

    def compose(self, other: Affine) -> Affine:
        """Return self ∘ other (apply other first, then self)."""
        return Affine(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f,
        )

    @classmethod
    def identity(cls) -> Affine:
        return cls()

    @classmethod
    def translate(cls, tx: float, ty: float) -> Affine:
        return cls(e=tx, f=ty)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Affine:
        sy = sx if sy is None else sy
        return cls(a=sx, d=sy)

    @classmethod
    def rotate(cls, degrees: float, cx: float = 0.0, cy: float = 0.0) -> Affine:
        rad = math.radians(degrees)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return (
            cls.translate(cx, cy)
            .compose(cls(a=cos_a, b=sin_a, c=-sin_a, d=cos_a))
            .compose(cls.translate(-cx, -cy))
        )


@dataclass(frozen=True, slots=True)
class Timing:
    start: float  # seconds
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


# ---------------------------------------------------------------------------
# Rigging / joints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoneDef:
    """One joint in a character skeleton."""

    id: str
    parent_id: str | None  # None = root
    pivot_x: float  # local pivot in SVG coords
    pivot_y: float
    layer_id: str | None = None  # SVG group / file this bone drives
    min_angle_deg: float = -180.0
    max_angle_deg: float = 180.0
    length: float = 0.0  # optional, for IK later


class MovementRuleType(str, Enum):
    WALK = "walk"
    JAW = "jaw"
    BOB = "bob"
    LOOK = "look"
    IDLE = "idle"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class MovementRule:
    """Declarative movement behaviour attached to a character."""

    type: MovementRuleType
    params: Mapping[str, float | str | bool] = field(default_factory=dict)
    # e.g. walk: {"stride": 18, "cycle": 0.6, "bob_amp": 6}
    #      jaw:  {"bone": "jaw", "scale": 1.0}
    #      bob:  {"bone": "head", "amp": 4, "freq": 2.5}


# ---------------------------------------------------------------------------
# Camera + backgrounds (parallax)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraState:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class BackgroundLayer:
    """One parallax layer."""

    path: Path
    z_index: int = 0
    parallax: float = 1.0  # 0 = fixed, 1 = moves with camera, >1 = foreground
    scroll_x: float = 0.0  # extra continuous scroll speed (px/s)
    scroll_y: float = 0.0
    repeat_x: bool = False
    repeat_y: bool = False
