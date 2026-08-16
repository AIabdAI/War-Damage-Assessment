#!/usr/bin/env python3
"""Evaluate a trained detector on the HELD-OUT TEST split (Approach finalists).

Part of the separate evaluation pipeline (evaluation/dvc.yaml) - deliberately
NOT in the training pipeline, so the test set is only touched on purpose:

    cd evaluation && dvc repro --single-item

The run name encodes the scheme (variant 11 or 22): e.g. yolo12m_det11.

Usage:
  python scripts/evaluate_detection.py --run yolo12m_det11 [--split test]

Outputs:
  reports/test_metrics_det_<run>.json    overall + per-class test metrics
  reports/figures/test_<run>/            confusion matrix + PR curve (test)
  mlflow.db                              experiment "war-damage-test-eval"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

EXPERIMENT = "war-damage-test-eval"
FIGURES = ("confusion_matrix_normalized.png", "confusion_matrix.png",
           "BoxPR_curve.png", "val_batch0_pred.jpg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True,
                        help="training run name, e.g. yolo12m_det11 / yolo26n_det22")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args(argv)

    variant = 22 if args.run.endswith("det22") else 11
    weights = Path("runs_detection") / args.run / "weights" / "best.pt"
    if not weights.is_file():
        print(f"ERROR: {weights} not found - dvc pull / train first", file=sys.stderr)
        return 2

    dataset_dir = Path(f"data/processed/detection{variant}")
    if not (dataset_dir / "data.yaml").is_file():
        print(f"ERROR: {dataset_dir}/data.yaml not found - run prepare_split first",
              file=sys.stderr)
        return 2
    # data.yaml carries the absolute path of whichever machine built it -
    # rewrite for THIS machine into a runtime copy (same pattern as training)
    data_cfg = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    data_cfg["path"] = dataset_dir.resolve().as_posix()
    data_yaml = Path("runs_detection") / f"data_{variant}.local.yaml"
    data_yaml.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding="utf-8")

    from ultralytics import YOLO

    try:
        import mlflow
    except ImportError:
        mlflow = None
        print("mlflow not installed - eval will not be tracked", file=sys.stderr)

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        plots=True,
        project=str(Path("runs_evaluation").resolve()),
        name=f"{args.run}_{args.split}",
        exist_ok=True,
        verbose=True,
    )

    md = getattr(results, "results_dict", {}) or {}
    payload = {
        "run": args.run,
        "variant": variant,
        "split": args.split,
        "mAP50": round(float(md.get("metrics/mAP50(B)", -1.0)), 4),
        "mAP50_95": round(float(md.get("metrics/mAP50-95(B)", -1.0)), 4),
        "precision": round(float(md.get("metrics/precision(B)", -1.0)), 4),
        "recall": round(float(md.get("metrics/recall(B)", -1.0)), 4),
    }
    p, r = payload["precision"], payload["recall"]
    payload["f1"] = round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0

    # per-class metrics (defensive: ultralytics metric internals shift between
    # versions - overall numbers above must survive any API drift here)
    try:
        box = results.box
        names = results.names
        per_class = {}
        for idx, ci in enumerate(box.ap_class_index):
            per_class[str(names[int(ci)])] = {
                "ap50": round(float(box.ap50[idx]), 4),
                "precision": round(float(box.p[idx]), 4),
                "recall": round(float(box.r[idx]), 4),
            }
        payload["per_class"] = per_class
    except Exception as exc:
        print(f"WARNING: per-class extraction failed ({exc}) - overall only",
              file=sys.stderr)

    out_json = Path(f"reports/{args.split}_metrics_det_{args.run}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"metrics written: {out_json}")

    fig_dir = Path("reports/figures") / f"{args.split}_{args.run}"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True)
    run_dir = Path(getattr(results, "save_dir",
                           Path("runs_evaluation") / f"{args.run}_{args.split}"))
    copied = 0
    for fig in FIGURES:
        src = run_dir / fig
        if src.is_file():
            shutil.copy2(src, fig_dir / fig)
            copied += 1
    print(f"figures copied: {copied} -> {fig_dir}")

    if mlflow is not None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "sqlite:///mlflow.db"))
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"test-{args.run}"):
            mlflow.set_tags({"stage": "test-eval", "run": args.run,
                             "variant": str(variant), "split": args.split})
            mlflow.log_metrics({k: v for k, v in payload.items()
                                if isinstance(v, float)})
            mlflow.log_artifact(str(out_json))
            if copied:
                mlflow.log_artifacts(str(fig_dir), artifact_path="figures")

    print(f"TEST RESULT {args.run}: mAP50={payload['mAP50']} "
          f"P={payload['precision']} R={payload['recall']} F1={payload['f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
