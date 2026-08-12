#!/usr/bin/env python3
"""Build a clean SMOKE detection dataset from only problem-free pairs.

Selection rule (a pair enters the dataset only if ALL hold):
  - label file parses: every line has exactly 6 columns, class_id in
    [0, num_classes-1], coords in [0, 1], damage_flag in {0, 1}
    (empty label file = valid negative sample)
  - a matching image exists in --raw-dir
  - the image decodes and is exactly 640x640

Everything else is EXCLUDED and listed with its reason in
<out>/excluded_report.txt. Old-scheme remnants (class_id >= 12) are excluded
automatically by the class-range rule.

Output (regenerable, safe to delete anytime):
  <out>/
    images/{train,val}/          copied images
    labels/{train,val}/          YOLO 5-column labels (damage flag stripped)
    data.yaml                    Ultralytics dataset file (12 classes)
    excluded_report.txt

Usage:
  python prepare_smoke_dataset.py                       # 300 imgs, 85/15 split
  python prepare_smoke_dataset.py --max-images 0        # ALL clean pairs
  python prepare_smoke_dataset.py --exclude-prefixes VID_,Crack-,concrete-
      # ^ recommended: drops old-name files whose class IDs <= 11 may still
      #   carry OLD-scheme meanings (semantic mixing the validator can't see)
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc

log = logging.getLogger("prepare_smoke")

IMG_EXTS = (".jpg", ".jpeg", ".png")
FALLBACK_CLASS_NAMES = [
    "Brick_Wall", "Column", "Staircase", "Floor_Tiles", "Sink",
    "Wall_Cabinet", "Window", "Door", "Air_Conditioner", "Light_Fixture",
    "Toilet",
]


def load_class_names(params_path: Path = Path("params.yaml")) -> list[str]:
    """Prefer params.yaml (single source of truth); fall back to constants."""
    try:
        import yaml
        names = yaml.safe_load(params_path.read_text(encoding="utf-8"))["classes"]["names"]
        ordered = [names[i] for i in sorted(names)]
        log.info("class names loaded from %s (%d classes)", params_path, len(ordered))
        return ordered
    except Exception:
        log.info("params.yaml not found/usable - using built-in 12-class list")
        return FALLBACK_CLASS_NAMES


def parse_clean_label(path: Path, num_classes: int) -> tuple[list[str] | None, str]:
    """Return (detection_lines, "") if fully clean, else (None, reason)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    det_lines: list[str] = []
    for i, line in enumerate(
            (ln.strip() for ln in text.replace("\r", "\n").split("\n")), start=1):
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 6:
            return None, f"line {i}: {len(tokens)} columns"
        try:
            cls = int(tokens[0])
            coords = [float(t) for t in tokens[1:5]]
            flag = int(tokens[5])
        except ValueError:
            return None, f"line {i}: non-numeric"
        if not 0 <= cls < num_classes:
            return None, f"line {i}: class_id {cls} outside [0,{num_classes - 1}]"
        if any(not 0.0 <= v <= 1.0 for v in coords):
            return None, f"line {i}: coordinate outside [0,1]"
        if flag not in (0, 1):
            return None, f"line {i}: damage_flag {flag}"
        det_lines.append(" ".join(tokens[:5]))
    return det_lines, ""


def find_image(raw_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = raw_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def image_ok(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            size = img.size
            img.verify()
    except (UnidentifiedImageError, OSError):
        return False, "image not decodable"
    if size != (640, 640):
        return False, f"image {size[0]}x{size[1]} (not 640x640)"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, default=Path("data/annotations/labels"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/smoke/detection"))
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=300,
                        help="cap for a fast smoke run; 0 = keep ALL clean pairs")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-prefixes", type=str, default="",
                        help="comma-separated stem prefixes to skip, e.g. VID_,Crack-")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.labels_dir.is_dir() or not args.raw_dir.is_dir():
        log.error("labels/raw dir not found - run from the repository root")
        return 2

    class_names = load_class_names()
    prefixes = tuple(p for p in (s.strip() for s in args.exclude_prefixes.split(",")) if p)

    clean: list[tuple[str, Path, list[str]]] = []
    excluded: list[tuple[str, str]] = []

    for lbl in sorted(args.labels_dir.glob("*.txt")):
        stem = lbl.stem
        if prefixes and stem.startswith(prefixes):
            excluded.append((stem, "excluded by prefix (possible old-scheme file)"))
            continue
        det_lines, reason = parse_clean_label(lbl, args.num_classes)
        if det_lines is None:
            excluded.append((stem, reason))
            continue
        img = find_image(args.raw_dir, stem)
        if img is None:
            excluded.append((stem, "no matching image"))
            continue
        ok, why = image_ok(img)
        if not ok:
            excluded.append((stem, why))
            continue
        clean.append((stem, img, det_lines))

    log.info("clean pairs: %d | excluded: %d", len(clean), len(excluded))
    if not clean:
        log.error("no clean pairs - nothing to prepare")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(clean)
    if args.max_images > 0:
        clean = clean[: args.max_images]

    n_val = max(1, int(round(len(clean) * args.val_ratio)))
    splits = {"val": clean[:n_val], "train": clean[n_val:]}
    if not splits["train"]:
        log.error("too few pairs for a train/val split (%d)", len(clean))
        return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    for split, items in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, img, det_lines in items:
            shutil.copy2(img, args.out / "images" / split / img.name)
            (args.out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(det_lines) + ("\n" if det_lines else ""),
                encoding="utf-8", newline="\n")
        log.info("%s: %d images", split, len(items))

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (args.out / "data.yaml").write_text(
        f"# generated by prepare_smoke_dataset.py - regenerable, machine-local\n"
        f"path: {args.out.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\n\nnames:\n{names_block}\n",
        encoding="utf-8", newline="\n")

    report = args.out / "excluded_report.txt"
    report.write_text(
        "\n".join(f"{s}\t{r}" for s, r in excluded) + "\n" if excluded else "none\n",
        encoding="utf-8", newline="\n")

    log.info("dataset ready: %s", args.out / "data.yaml")
    log.info("excluded pairs listed in: %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
