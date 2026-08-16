#!/usr/bin/env python3
"""Evaluate a trained damage classifier on the HELD-OUT TEST crops.

Part of the separate evaluation pipeline (evaluation/dvc.yaml). Loads the
checkpoint saved by train_classifier.py (payload: backbone, classes, imgsz,
state_dict) and scores data/processed/classification/<split>/.

Usage:
  python scripts/evaluate_classifier.py --model swin [--split test]

Outputs:
  reports/test_metrics_cls_<model>.json
  reports/figures/test_cls_<model>/confusion_matrix.png
  mlflow.db   experiment "war-damage-test-eval"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPERIMENT = "war-damage-test-eval"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        choices=["dinov2", "efficientnet", "swin"])
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)

    ckpt_path = Path("runs_classification") / args.model / "weights" / "best.pt"
    data_dir = Path("data/processed/classification") / args.split
    if not ckpt_path.is_file():
        print(f"ERROR: {ckpt_path} not found", file=sys.stderr)
        return 2
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} not found - run crop_classification first",
              file=sys.stderr)
        return 2

    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    try:
        import mlflow
    except ImportError:
        mlflow = None
        print("mlflow not installed - eval will not be tracked", file=sys.stderr)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone, imgsz = ckpt["backbone"], int(ckpt["imgsz"])
    model_kw = {"pretrained": False, "num_classes": 2}
    if "patch14" in backbone:
        model_kw["img_size"] = imgsz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(backbone, **model_kw)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(int(round(imgsz / 0.875))),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tf)
    assert ds.classes == ["damaged", "undamaged"], ds.classes
    pos_idx = ds.class_to_idx["damaged"]
    dl = DataLoader(ds, batch_size=args.batch, num_workers=2,
                    pin_memory=device.type == "cuda")

    tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            tp += int(((pred == pos_idx) & (y == pos_idx)).sum())
            fp += int(((pred == pos_idx) & (y != pos_idx)).sum())
            fn += int(((pred != pos_idx) & (y == pos_idx)).sum())
            tn += int(((pred != pos_idx) & (y != pos_idx)).sum())

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    payload = {
        "model": args.model, "backbone": backbone, "split": args.split,
        "crops": total,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision_damaged": round(precision, 4),
        "recall_damaged": round(recall, 4),
        "f1_damaged": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    out_json = Path(f"reports/{args.split}_metrics_cls_{args.model}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"metrics written: {out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path("reports/figures") / f"{args.split}_cls_{args.model}"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True)
    cm = [[tp, fn], [fp, tn]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["pred damaged", "pred undamaged"])
    ax.set_yticks([0, 1], ["true damaged", "true undamaged"])
    ax.set_title(f"{args.model} - {args.split} split ({total} crops)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    if mlflow is not None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "sqlite:///mlflow.db"))
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"test-cls-{args.model}"):
            mlflow.set_tags({"stage": "test-eval", "model": args.model,
                             "split": args.split})
            mlflow.log_metrics({k: v for k, v in payload.items()
                                if isinstance(v, float)})
            mlflow.log_artifact(str(out_json))
            mlflow.log_artifacts(str(fig_dir), artifact_path="figures")

    print(f"TEST RESULT cls-{args.model}: acc={payload['accuracy']} "
          f"F1(damaged)={payload['f1_damaged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
