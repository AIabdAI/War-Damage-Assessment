#!/usr/bin/env python3
"""One-time fix: clamp YOLO boxes that spill past the image edge.

2026-08-09: the Phase-0 gate found 23 label files (tile_*, bricks_*) whose
box w or h slightly exceeds 1.0 (max 1.037) - float artifacts from tiling.
The part beyond the edge is outside the frame, so we clip the box edges to
[0, 1] and recompute center/size. Only offending lines are rewritten;
everything else is preserved byte-for-byte.

Usage:  python scripts/clamp_label_boxes.py [--labels data/annotations/labels]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def clamp_line(tokens: list[str]) -> list[str] | None:
    """Return fixed tokens if the box needs clamping, else None."""
    cls, cx, cy, w, h, flag = (tokens[0], *map(float, tokens[1:5]), tokens[5])
    left, right = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    top, bottom = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    ncx, nw = (left + right) / 2, right - left
    ncy, nh = (top + bottom) / 2, bottom - top
    if (ncx, ncy, nw, nh) == (cx, cy, w, h):
        return None
    return [cls] + [f"{v:.6f}" for v in (ncx, ncy, nw, nh)] + [flag]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/annotations/labels"))
    args = parser.parse_args()

    fixed_files = 0
    for path in sorted(args.labels.glob("*.txt")):
        if path.name.startswith("."):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        for i, line in enumerate(lines):
            tokens = line.split()
            if len(tokens) != 6:
                continue
            try:
                coords = [float(t) for t in tokens[1:5]]
            except ValueError:
                continue
            # fix only hard gate errors (coordinate value outside [0,1]);
            # box-EXTENT overflow is a warning the cropper handles - leave it
            if any(not 0.0 <= v <= 1.0 for v in coords):
                new_tokens = clamp_line(tokens)
                if new_tokens:
                    lines[i] = " ".join(new_tokens)
                    changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            fixed_files += 1
            print(f"clamped: {path.name}")

    print(f"\n{fixed_files} file(s) fixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
