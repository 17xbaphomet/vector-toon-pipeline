"""Load CharacterRig from a folder layout under assets/characters/<id>/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from domain.entities import CharacterRig
from domain.interfaces import CharacterAssetRepository
from domain.value_objects import Viseme


class FileCharacterAssetRepository(CharacterAssetRepository):
    """
    Expected layout:

    assets/characters/<id>/
        character.json
        body.svg
        mouths/
            X.svg  A.svg  B.svg  ... H.svg
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list_ids(self) -> Sequence[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "character.json").exists()
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

        layer_paths = {"body": body}
        for name, rel in meta.get("layers", {}).items():
            layer_paths[name] = base / rel

        return CharacterRig(
            id=character_id,
            base_svg=body,
            layer_paths=layer_paths,
            mouth_shapes=mouth_shapes,
            bone_order=tuple(meta.get("bone_order", [])),
            default_scale=float(meta.get("default_scale", 1.0)),
        )
