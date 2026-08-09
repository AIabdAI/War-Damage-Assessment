#!/usr/bin/env python3
"""Quick SMOKE detection training on the prepared clean subset.

Purpose: verify the whole plumbing (dataset format, labels, training loop)
in minutes - NOT to produce a good model. Full training happens on the A100
via the pipeline (Phase 6).

Config comes from params.yaml (train_smoke: epochs, imgsz, models) so that
`dvc repro` reacts to parameter changes. CLI flags override params.yaml.

Usage (after prepare_smoke_dataset.py):
  python train_smoke.py                    # all models from params.yaml
  python train_smoke.py --model yolo26n --epochs 5
  python train_smoke.py --batch 4          # if RAM is tight

Outputs:
  runs_smoke/<model-name>/     weights, curves, confusion matrix
  reports/metrics.json         per-model metrics (DVC metrics file)
  mlflow.db                    one MLflow run per model (experiment: war-damage-smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

METRIC_KEYS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def load_params(path: Path = Path("params.yaml")) -> dict:
    """train_smoke section of params.yaml; empty dict if unavailable."""
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))["train_smoke"]
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path,
                        default=Path("data/processed/smoke/detection/data.yaml"))
    parser.add_argument("--model", type=str, default=None,
                        help="train a single model instead of the params.yaml list")
    parser.add_argument("--epochs", type=int, default=params.get("epochs", 3))
    parser.add_argument("--imgsz", type=int, default=params.get("imgsz", 640))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    models = [args.model] if args.model else list(params.get("models", ["yolo12n"]))

    if not args.data.is_file():
        print(f"dataset file not found: {args.data}\n"
              "run prepare_smoke_dataset.py first", file=sys.stderr)
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("missing deps - install with:  pip install -r requirements-train.txt",
              file=sys.stderr)
        return 2

    try:
        import mlflow
    except ImportError:
        mlflow = None
        print("mlflow not installed - runs will not be tracked", file=sys.stderr)
    else:
        # steer the built-in ultralytics MLflow callback (it logs args +
        # per-epoch metrics); keep the run active so we can append the
        # baseline-compatible summary metrics below, then close it ourselves
        os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "war-damage-smoke"
        os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"

    device = 0 if torch.cuda.is_available() else "cpu"
    all_metrics: dict[str, dict] = {}

    for model_name in models:
        weights = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        stem = Path(weights).stem
        print(f"\ndevice: {device} | model: {weights} | epochs: {args.epochs}")
        if mlflow is not None:
            os.environ["MLFLOW_RUN"] = f"smoke-{stem}"

        results = YOLO(weights).train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            device=device,
            workers=2,
            # absolute path: ultralytics nests a relative project under
            # its settings runs_dir (runs/detect/...) instead of CWD
            project=str(Path("runs_smoke").resolve()),
            name=stem,
            exist_ok=True,
            verbose=True,
            plots=True,
        )

        md = getattr(results, "results_dict", {}) or {}
        all_metrics[stem] = {
            "epochs": args.epochs,
            **{k: round(float(md.get(v, -1.0)), 4) for k, v in METRIC_KEYS.items()},
        }
        print(f"=== SMOKE RESULT {stem} ===")
        for k, v in all_metrics[stem].items():
            print(f"  {k}: {v}")

        # append summary metrics named like the registered baselines
        # (mAP50, mAP50_95) into the run the ultralytics callback opened
        if mlflow is not None and mlflow.active_run() is not None:
            mlflow.log_metrics({k: v for k, v in all_metrics[stem].items()
                                if k != "epochs"})
            mlflow.end_run()

    # metrics for DVC (dvc.yaml: metrics -> reports/metrics.json)
    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nmetrics written: {metrics_path}")
    print("smoke training completed - the plumbing works.")
    return 0


if __name__ == "__main__":       # required on Windows (spawn-safe dataloaders)
    sys.exit(main())
