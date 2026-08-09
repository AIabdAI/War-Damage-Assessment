#!/usr/bin/env python3
"""Quick SMOKE detection training on the prepared clean subset.

Purpose: verify the whole plumbing (dataset format, labels, training loop)
in minutes - NOT to produce a good model. Full training happens on the A100
via the pipeline (Phase 6).

Usage (after prepare_smoke_dataset.py):
  python train_smoke.py                              # yolo12n, 3 epochs
  python train_smoke.py --model yolo26n.pt --epochs 5
  python train_smoke.py --batch 4                    # if RAM is tight

Requires:  pip install ultralytics
Outputs:   runs_smoke/<model-name>/  (weights, curves, confusion matrix)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path,
                        default=Path("data/processed/smoke/detection/data.yaml"))
    parser.add_argument("--model", type=str, default="yolo12n.pt",
                        help="yolo12n.pt or yolo26n.pt (auto-downloaded)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print(f"dataset file not found: {args.data}\n"
              "run prepare_smoke_dataset.py first", file=sys.stderr)
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("missing deps - install with:  pip install ultralytics",
              file=sys.stderr)
        return 2

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {args.model} | epochs: {args.epochs}")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=device,
        workers=2,
        project="runs_smoke",
        name=Path(args.model).stem,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # final validation metrics (also saved under runs_smoke/<name>/)
    md = getattr(results, "results_dict", {}) or {}
    print("\n=== SMOKE RESULT ===")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "metrics/precision(B)", "metrics/recall(B)"):
        if key in md:
            print(f"  {key}: {md[key]:.4f}")
    print(f"  artifacts: runs_smoke/{Path(args.model).stem}/")
    print("smoke training completed - the plumbing works.")
    return 0


if __name__ == "__main__":       # required on Windows (spawn-safe dataloaders)
    sys.exit(main())
