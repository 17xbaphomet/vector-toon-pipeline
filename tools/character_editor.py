#!/usr/bin/env python3
"""Streamlit Character Editor – joints & movement rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.value_objects import BoneDef, MovementRule, MovementRuleType
from domain.entities import CharacterRig
from infrastructure.assets.file_repo import FileCharacterAssetRepository

REPO = FileCharacterAssetRepository(ROOT / "assets" / "characters")

st.set_page_config(page_title="Vector-Toon Character Editor", layout="wide")
st.title("Character Editor – Joints & Movement Rules")

ids = list(REPO.list_ids())
choice = st.sidebar.selectbox("Character", ["(new)"] + ids)

if choice == "(new)":
    new_id = st.sidebar.text_input("New character id", value="hero")
    if st.sidebar.button("Create"):
        (ROOT / "assets" / "characters" / new_id).mkdir(parents=True, exist_ok=True)
        st.rerun()
    st.info("Create a character folder, drop body.svg into it, then reload.")
    st.stop()

rig = REPO.load(choice)
st.sidebar.success(f"Loaded **{rig.id}**")

st.header("Bones / Joints")
bones_data = []
for i, b in enumerate(rig.bones or []):
    cols = st.columns([2, 2, 1, 1, 1, 1, 1])
    with cols[0]:
        bid = st.text_input("id", b.id, key=f"id_{i}")
    with cols[1]:
        parent = st.text_input("parent", b.parent_id or "", key=f"par_{i}")
    with cols[2]:
        px = st.number_input("pivot_x", value=float(b.pivot_x), key=f"px_{i}")
    with cols[3]:
        py = st.number_input("pivot_y", value=float(b.pivot_y), key=f"py_{i}")
    with cols[4]:
        amin = st.number_input("min°", value=float(b.min_angle_deg), key=f"amin_{i}")
    with cols[5]:
        amax = st.number_input("max°", value=float(b.max_angle_deg), key=f"amax_{i}")
    with cols[6]:
        layer = st.text_input("layer", b.layer_id or "", key=f"lay_{i}")
    bones_data.append(BoneDef(id=bid, parent_id=parent or None, pivot_x=px, pivot_y=py,
                              layer_id=layer or None, min_angle_deg=amin, max_angle_deg=amax))

if not bones_data:
    st.info("No bones yet – edit character.json or add via code.")

st.header("Movement Rules")
rules_data = []
for i, r in enumerate(rig.rules or []):
    c1, c2 = st.columns([1, 3])
    with c1:
        opts = [t.value for t in MovementRuleType]
        idx = opts.index(r.type.value) if r.type.value in opts else 0
        rtype = st.selectbox("type", opts, index=idx, key=f"rtype_{i}")
    with c2:
        params_str = st.text_input("params (JSON)", value=json.dumps(dict(r.params)), key=f"rparams_{i}")
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        params = dict(r.params)
    rules_data.append(MovementRule(type=MovementRuleType(rtype), params=params))

st.header("Base SVG")
if rig.base_svg.is_file():
    st.image(str(rig.base_svg), width=300)

if st.button("💾 Save character.json", type="primary"):
    updated = CharacterRig(
        id=rig.id, base_svg=rig.base_svg, layer_paths=rig.layer_paths,
        mouth_shapes=rig.mouth_shapes,
        bone_order=tuple(b.id for b in bones_data) or rig.bone_order,
        bones=tuple(bones_data), rules=tuple(rules_data),
        default_scale=rig.default_scale,
    )
    REPO.save(updated)
    st.success(f"Saved assets/characters/{rig.id}/character.json")
