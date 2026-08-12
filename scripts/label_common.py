#!/usr/bin/env python3
"""Canonical label parsing shared by prepare_split and crop_classification.

Both dataset builders MUST see exactly the same objects with exactly the same
geometry and 1-based line numbers: crops are named by line number and must
stay aligned with the detection labels built from the same file. Any change
here affects both stages together - never fork this logic into the scripts.

Rules implemented (the single source of truth):
- line iteration: text.replace("\\r", "\\n").split("\\n"), 1-based numbering
- skip reasons: missing_damage_flag (5 tokens), bad_columns (!=6 tokens),
  non_numeric, old_scheme_remnant (cls outside [0, num_classes)),
  invalid_damage_flag (flag not in {0, 1})
- coords outside [0, 1]: box edges clamped into [0, 1] (recorded, not skipped)
- image lookup: case-insensitive extension match, duplicate stems resolved by
  IMG_EXTS priority then filename (deterministic across Windows and Linux CI)
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")   # priority order for duplicate stems
SPLITS = ("train", "val", "test")
LINE_REASONS = ("missing_damage_flag", "bad_columns", "non_numeric",
                "old_scheme_remnant", "invalid_damage_flag")


@dataclass
class LabelObject:
    """One valid (possibly clamped) annotation line."""
    line: int  # 1-based line number in the source label file
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    flag: int


def clamp_axis(center: float, size: float) -> tuple[float, float]:
    """Clip box edges to [0, 1] on one axis and recompute center/size."""
    lo = max(0.0, center - size / 2)
    hi = min(1.0, center + size / 2)
    return (lo + hi) / 2, hi - lo


def parse_label_file(
    path: Path, num_classes: int,
) -> tuple[list[LabelObject], list[dict], list[dict]]:
    """Apply the canonical parse/skip/clamp rules to every line of one file.

    Returns (valid_objects, bad_line_records, clamped_line_records); records
    are {"file": <label filename>, "line": <1-based>, "reason": <LINE_REASONS>}
    (clamped records carry no reason - they are informational, not skips).
    """
    objects: list[LabelObject] = []
    bad: list[dict] = []
    clamped: list[dict] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, raw in enumerate(text.replace("\r", "\n").split("\n"), start=1):
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 6:
            reason = "missing_damage_flag" if len(tokens) == 5 else "bad_columns"
            bad.append({"file": path.name, "line": i, "reason": reason})
            continue
        try:
            cls = int(tokens[0])
            cx, cy, w, h = (float(t) for t in tokens[1:5])
            flag = int(tokens[5])
        except ValueError:
            bad.append({"file": path.name, "line": i, "reason": "non_numeric"})
            continue
        if not 0 <= cls < num_classes:
            bad.append({"file": path.name, "line": i, "reason": "old_scheme_remnant"})
            continue
        if flag not in (0, 1):
            bad.append({"file": path.name, "line": i, "reason": "invalid_damage_flag"})
            continue
        if any(not 0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            cx, w = clamp_axis(cx, w)
            cy, h = clamp_axis(cy, h)
            clamped.append({"file": path.name, "line": i})
        objects.append(LabelObject(i, cls, cx, cy, w, h, flag))
    return objects, bad, clamped


def index_images(raw_dir: Path) -> dict[str, Path]:
    """Map image stem -> path for every image in the flat raw dir.

    Deterministic on every OS: extensions matched case-insensitively (a Linux
    CI run must agree with Windows), duplicate stems resolved by IMG_EXTS
    priority then filename.
    """
    candidates: dict[str, list[Path]] = {}
    for p in raw_dir.iterdir():
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMG_EXTS:
            candidates.setdefault(p.stem, []).append(p)
    return {
        stem: min(paths, key=lambda p: (IMG_EXTS.index(p.suffix.lower()), p.name))
        for stem, paths in candidates.items()
    }


def place_image(src: Path, dst: Path) -> None:
    """Hardlink when possible (same volume), else copy. Source never modified."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
