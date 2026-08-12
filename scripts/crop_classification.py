#!/usr/bin/env python3
"""Crop labeled objects into the damaged/undamaged classification dataset.

Reads the split computed by prepare_split (reports/split_manifest.json), the
ORIGINAL 6-column YOLO labels ("cls cx cy w h damage_flag") from
data/annotations/labels and the ORIGINAL images from data/raw, then writes
padded JPEG crops to

    data/processed/classification/{train,val,test}/{damaged,undamaged}/
        {image_stem}_{line:03d}_{ClassName}.jpg

Label lines are parsed with the SAME canonical skip + clamp rules as
prepare_split (so object line numbers stay aligned across detection variants
and crops). Empty label files are valid negatives and simply produce no crops.

Outputs (besides the crops):
    reports/crop_skipped.json          per-line skips + reason counts
    reports/classification_stats.json  damaged/undamaged counts per split/class

Usage:
    python scripts/crop_classification.py [--params params.yaml]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc

try:  # run as `python scripts/crop_classification.py` (sys.path[0] = scripts/)
    from label_common import (LINE_REASONS, SPLITS, index_images,
                              parse_label_file, write_json)
except ImportError:  # imported from the repository root (tests)
    from scripts.label_common import (LINE_REASONS, SPLITS, index_images,
                                      parse_label_file, write_json)

SKIP_REASONS = LINE_REASONS + ("image_missing", "too_small")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()

    if not args.params.is_file():
        print(f"ERROR: {args.params} not found - run from the repository root",
              file=sys.stderr)
        return 2
    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in params["classes"]["names"].items()}
    num_classes = int(params["classes"]["num_classes"])
    padding_pct = float(params["crop"]["padding_pct"])
    min_crop_px = int(params["crop"]["min_crop_px"])
    data_cfg = params.get("data", {})
    labels_dir = Path(data_cfg.get("labels", "data/annotations/labels"))
    raw_dir = Path(data_cfg.get("raw_images", "data/raw"))
    out_root = Path(data_cfg.get("processed", "data/processed")) / "classification"

    manifest_path = Path("reports/split_manifest.json")
    if not manifest_path.is_file():
        print(
            f"ERROR: {manifest_path} not found - run prepare_split first "
            "(crop_classification never re-splits).",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[str, list[str]] = manifest["splits"]

    images_by_stem = index_images(raw_dir)

    # Fresh DVC stage output; raw data and annotations are never touched.
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in SPLITS:
        for bucket in ("damaged", "undamaged"):
            (out_root / split / bucket).mkdir(parents=True, exist_ok=True)

    class_names_ordered = [names[i] for i in sorted(names)]
    stats: dict[str, dict] = {
        split: {
            "damaged": 0,
            "undamaged": 0,
            "per_class": {n: {"damaged": 0, "undamaged": 0} for n in class_names_ordered},
        }
        for split in SPLITS
    }
    skip_records: list[dict] = []
    skip_counts: dict[str, int] = {reason: 0 for reason in SKIP_REASONS}

    def log_skip(label_name: str, line_no: int, reason: str) -> None:
        skip_records.append({"file": label_name, "line": line_no, "reason": reason})
        skip_counts[reason] += 1

    for split in SPLITS:
        for stem in splits.get(split, []):
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.is_file():
                print(f"WARNING: label file missing for manifest stem '{stem}' - skipped")
                continue
            valid, bad, _clamped = parse_label_file(label_path, num_classes)
            for rec in bad:
                log_skip(rec["file"], rec["line"], rec["reason"])
            if not valid:  # negative (empty label) or nothing usable: no crops
                continue

            img_path = images_by_stem.get(stem)
            if img_path is None:
                log_skip(label_path.name, 0, "image_missing")
                continue
            # decode failures only; save-time errors (disk full, permission)
            # must propagate, not masquerade as "image_missing"
            try:
                img = Image.open(img_path)
                img.load()
            except (UnidentifiedImageError, OSError):
                log_skip(label_path.name, 0, "image_missing")
                continue
            with img:
                img_w, img_h = img.size
                for o in valid:
                    w_px, h_px = o.w * img_w, o.h * img_h
                    left = (o.cx - o.w / 2) * img_w - w_px * padding_pct
                    right = (o.cx + o.w / 2) * img_w + w_px * padding_pct
                    top = (o.cy - o.h / 2) * img_h - h_px * padding_pct
                    bottom = (o.cy + o.h / 2) * img_h + h_px * padding_pct
                    x0, y0 = int(max(0.0, left)), int(max(0.0, top))
                    x1 = int(min(float(img_w), right))
                    y1 = int(min(float(img_h), bottom))
                    if x1 - x0 < min_crop_px or y1 - y0 < min_crop_px:
                        log_skip(label_path.name, o.line, "too_small")
                        continue
                    bucket = "damaged" if o.flag == 1 else "undamaged"
                    crop_name = f"{stem}_{o.line:03d}_{names[o.cls]}.jpg"
                    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
                    crop.save(out_root / split / bucket / crop_name,
                              "JPEG", quality=95)
                    stats[split][bucket] += 1
                    stats[split]["per_class"][names[o.cls]][bucket] += 1

    write_json(Path("reports/crop_skipped.json"),
               {"skipped": skip_records, "counts": skip_counts})
    write_json(Path("reports/classification_stats.json"), stats)

    for split in SPLITS:
        total = stats[split]["damaged"] + stats[split]["undamaged"]
        print(f"{split}: {total} crops "
              f"({stats[split]['damaged']} damaged, "
              f"{stats[split]['undamaged']} undamaged)")
    print(f"skipped lines: {sum(skip_counts.values())} "
          f"({', '.join(f'{k}={v}' for k, v in skip_counts.items() if v)})"
          if any(skip_counts.values()) else "skipped lines: 0")
    print(f"dataset: {out_root.resolve().as_posix()}")
    print("reports: reports/crop_skipped.json, reports/classification_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
