"""Load / save CharacterRig from assets/characters/<id>/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from domain.entities import CharacterRig
from domain.interfaces import CharacterAssetRepository
from domain.value_objects import BoneDef, MovementRule, MovementRuleType, Viseme


class FileCharacterAssetRepository(CharacterAssetRepository):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list_ids(self) -> Sequence[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / "character.json").exists()
        )

    def load(self, character_id: str) -> CharacterRig:
        base = self.root / character_id
        meta_path = base / "character.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No character.json for '{character_id}' at {base}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        body = base / meta.get("base_svg", "body.svg")

        mouths_dir = base / "mouths"
        mouth_shapes: dict[Viseme, Path] = {}
        for v in Viseme:
            candidate = mouths_dir / f"{v.value}.svg"
            if candidate.is_file():
                mouth_shapes[v] = candidate

        layer_paths: dict[str, Path] = {"body": body}
        for name, rel in meta.get("layers", {}).items():
            layer_paths[name] = base / rel

        bones = [
            BoneDef(
                id=b["id"],
                parent_id=b.get("parent_id"),
                pivot_x=float(b.get("pivot_x", 0)),
                pivot_y=float(b.get("pivot_y", 0)),
                layer_id=b.get("layer_id"),
                min_angle_deg=float(b.get("min_angle_deg", -180)),
                max_angle_deg=float(b.get("max_angle_deg", 180)),
                length=float(b.get("length", 0)),
            )
            for b in meta.get("bones", [])
        ]

        rules = []
        for r in meta.get("rules", []):
            try:
                rtype = MovementRuleType(r["type"])
            except ValueError:
                rtype = MovementRuleType.CUSTOM
            rules.append(MovementRule(type=rtype, params=r.get("params", {})))

        return CharacterRig(
            id=character_id,
            base_svg=body,
            layer_paths=layer_paths,
            mouth_shapes=mouth_shapes,
            bone_order=tuple(meta.get("bone_order", [b.id for b in bones])),
            bones=tuple(bones),
            rules=tuple(rules),
            default_scale=float(meta.get("default_scale", 1.0)),
        )

    def save(self, rig: CharacterRig) -> None:
        base = self.root / rig.id
        base.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": rig.id,
            "base_svg": rig.base_svg.name if isinstance(rig.base_svg, Path) else str(rig.base_svg),
            "default_scale": rig.default_scale,
            "bone_order": list(rig.bone_order),
            "layers": {
                k: str(v)
                for k, v in rig.layer_paths.items()
                if k != "body"
            },
            "bones": [
                {
                    "id": b.id,
                    "parent_id": b.parent_id,
                    "pivot_x": b.pivot_x,
                    "pivot_y": b.pivot_y,
                    "layer_id": b.layer_id,
                    "min_angle_deg": b.min_angle_deg,
                    "max_angle_deg": b.max_angle_deg,
                    "length": b.length,
                }
                for b in rig.bones
            ],
            "rules": [
                {"type": r.type.value, "params": dict(r.params)} for r in rig.rules
            ],
        }
        (base / "character.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
