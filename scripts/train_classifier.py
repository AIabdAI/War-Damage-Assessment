#!/usr/bin/env python3
"""Train a damaged/undamaged crop classifier (Approach B, stage 2).

Fine-tunes a pretrained timm backbone on the crops produced by
crop_classification (data/processed/classification/{train,val}/
{damaged,undamaged}). The three configured backbones (params.yaml
train_classifier.models):

    dinov2       vit_small_patch14_dinov2.lvd142m
    efficientnet efficientnet_b0
    swin         swin_tiny_patch4_window7_224

Usage:
  python scripts/train_classifier.py --model dinov2            # full run
  python scripts/train_classifier.py --model swin --smoke      # CI plumbing check
  python scripts/train_classifier.py --model efficientnet --epochs 5 --batch 16

Outputs:
  runs_classification/<model>/         weights/{best,last}.pt, results.csv, plots
  reports/metrics_cls_<model>.json     best-epoch metrics (DVC metrics file)
  reports/figures/cls_<model>/         curves + confusion matrix (CML report)
  mlflow.db                            experiment "war-damage-classification"

The positive class is "damaged". Metrics follow the best val F1(damaged) epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

EXPERIMENT = "war-damage-classification"
POSITIVE = "damaged"  # ImageFolder sorts alphabetically -> damaged = index 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_params(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["train_classifier"]


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision_damaged": precision,
        "recall_damaged": recall,
        "f1_damaged": f1,
    }


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--params", type=Path, default=Path("params.yaml"))
    pre_args, _ = pre.parse_known_args(argv)
    if not pre_args.params.is_file():
        print(f"ERROR: {pre_args.params} not found - run from the repository root",
              file=sys.stderr)
        return 2
    cfg = load_params(pre_args.params)
    model_names = list(cfg["models"])

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre])
    parser.add_argument("--model", choices=model_names, default=model_names[0])
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on a small fraction - plumbing check only")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args(argv)

    backbone = str(cfg["models"][args.model])
    imgsz = int(cfg["imgsz"])
    batch = args.batch or int(cfg["batch"])
    lr = args.lr or float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    seed = int(cfg["seed"])
    if args.smoke:
        epochs = int(cfg["smoke"]["epochs"])
        fraction = float(cfg["smoke"]["fraction"])
    else:
        epochs = args.epochs or int(cfg["epochs"])
        fraction = 1.0

    data_root = Path("data/processed/classification")
    if not (data_root / "train").is_dir() or not (data_root / "val").is_dir():
        print(f"ERROR: {data_root} not found - run crop_classification first",
              file=sys.stderr)
        return 2

    try:
        import timm
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError as exc:
        print(f"missing deps ({exc}) - pip install -r requirements-train.txt",
              file=sys.stderr)
        return 2

    try:
        import mlflow
    except ImportError:
        mlflow = None
        print("mlflow not installed - runs will not be tracked", file=sys.stderr)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- data ----------------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(round(imgsz / 0.875))),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=val_tf)
    assert train_ds.classes == [POSITIVE, "undamaged"], train_ds.classes
    pos_idx = train_ds.class_to_idx[POSITIVE]

    if fraction < 1.0:  # smoke: same seeded subsetting for train and val
        g = torch.Generator().manual_seed(seed)
        train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)
                          [:max(batch, int(len(train_ds) * fraction))].tolist())
        val_ds = Subset(val_ds, torch.randperm(len(val_ds), generator=g)
                        [:max(batch, int(len(val_ds) * fraction))].tolist())

    loader_kw = dict(batch_size=batch, num_workers=2, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > batch,
                          **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"device: {device} | model: {args.model} ({backbone}) | "
          f"epochs: {epochs} | train: {len(train_ds)} | val: {len(val_ds)} crops")

    # -- model / optimizer -----------------------------------------------------
    model_kw = {"pretrained": True, "num_classes": 2}
    if "patch14" in backbone:  # ViT/DINOv2: override the 518px pretrain size
        model_kw["img_size"] = imgsz
    model = timm.create_model(backbone, **model_kw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    run_dir = Path("runs_classification") / args.model
    if run_dir.exists():
        shutil.rmtree(run_dir)  # DVC stage out - fresh on every run
    (run_dir / "weights").mkdir(parents=True)

    # -- mlflow -----------------------------------------------------------------
    run = None
    if mlflow is not None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "sqlite:///mlflow.db"))
        mlflow.set_experiment(EXPERIMENT)
        run = mlflow.start_run(
            run_name=f"{args.model}-cls" + ("-smoke" if args.smoke else ""))
        mlflow.set_tags({"approach": "B", "stage": "classifier",
                         "model": args.model, "smoke": str(args.smoke)})
        mlflow.log_params({"backbone": backbone, "epochs": epochs, "imgsz": imgsz,
                           "batch": batch, "lr": lr, "weight_decay": weight_decay,
                           "seed": seed, "train_crops": len(train_ds),
                           "val_crops": len(val_ds), "positive_class": POSITIVE})

    best = {"f1_damaged": -1.0}
    history: list[dict] = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, seen = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * x.size(0)
                seen += x.size(0)
            scheduler.step()
            train_loss = running_loss / max(seen, 1)

            model.eval()
            tp = fp = fn = tn = 0
            val_loss, vseen = 0.0, 0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device.type,
                                            enabled=device.type == "cuda"):
                        logits = model(x)
                        val_loss += criterion(logits, y).item() * x.size(0)
                    vseen += x.size(0)
                    pred = logits.argmax(1)
                    tp += int(((pred == pos_idx) & (y == pos_idx)).sum())
                    fp += int(((pred == pos_idx) & (y != pos_idx)).sum())
                    fn += int(((pred != pos_idx) & (y == pos_idx)).sum())
                    tn += int(((pred != pos_idx) & (y != pos_idx)).sum())
            m = binary_metrics(tp, fp, fn, tn)
            row = {"epoch": epoch, "train_loss": round(train_loss, 5),
                   "val_loss": round(val_loss / max(vseen, 1), 5),
                   **{k: round(v, 4) for k, v in m.items()}}
            history.append(row)
            print(" | ".join(f"{k}={v}" for k, v in row.items()))
            if mlflow is not None:
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"},
                                   step=epoch)

            torch.save({"backbone": backbone, "classes": ["damaged", "undamaged"],
                        "imgsz": imgsz, "state_dict": model.state_dict()},
                       run_dir / "weights" / "last.pt")
            if m["f1_damaged"] >= best["f1_damaged"]:
                best = {**m, "epoch": epoch, "confusion":
                        {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}
                shutil.copy2(run_dir / "weights" / "last.pt",
                             run_dir / "weights" / "best.pt")

        # -- artifacts: results.csv, curves, confusion matrix -------------------
        with (run_dir / "results.csv").open("w", newline="",
                                            encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        xs = [r["epoch"] for r in history]
        ax1.plot(xs, [r["train_loss"] for r in history], label="train loss")
        ax1.plot(xs, [r["val_loss"] for r in history], label="val loss")
        ax1.set_xlabel("epoch"), ax1.legend(), ax1.set_title("loss")
        ax2.plot(xs, [r["f1_damaged"] for r in history], label="F1 (damaged)")
        ax2.plot(xs, [r["accuracy"] for r in history], label="accuracy")
        ax2.set_xlabel("epoch"), ax2.legend(), ax2.set_title("val metrics")
        fig.suptitle(f"{args.model} ({backbone})")
        fig.tight_layout()
        fig.savefig(run_dir / "results.png", dpi=120)
        plt.close(fig)

        c = best["confusion"]
        cm = [[c["tp"], c["fn"]], [c["fp"], c["tn"]]]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["pred damaged", "pred undamaged"])
        ax.set_yticks([0, 1], ["true damaged", "true undamaged"])
        ax.set_title(f"{args.model} - best epoch {best['epoch']}")
        fig.tight_layout()
        fig.savefig(run_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)

        fig_dir = Path("reports/figures") / f"cls_{args.model}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name in ("results.png", "confusion_matrix.png"):
            shutil.copy2(run_dir / name, fig_dir / name)

        metrics_path = Path(f"reports/metrics_cls_{args.model}.json")
        payload = {"model": args.model, "backbone": backbone, "smoke": args.smoke,
                   "epochs": epochs, "best_epoch": best["epoch"],
                   **{k: round(best[k], 4) for k in
                      ("accuracy", "precision_damaged", "recall_damaged",
                       "f1_damaged")}}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n",
                                encoding="utf-8")
        print(f"metrics written: {metrics_path}")

        if mlflow is not None:
            mlflow.log_metrics({f"best_{k}": best[k] for k in
                                ("accuracy", "precision_damaged",
                                 "recall_damaged", "f1_damaged")})
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifacts(str(fig_dir), artifact_path="figures")
            mlflow.log_artifact(str(run_dir / "weights" / "best.pt"),
                                artifact_path="weights")
            try:
                # MLflow 3: runs:/ URIs resolve to logged-model entities,
                # so register the raw artifact via create_model_version
                reg_name = f"{args.model}-cls"
                client = mlflow.MlflowClient()
                try:
                    client.create_registered_model(reg_name)
                except Exception:
                    pass  # already exists
                mv = client.create_model_version(
                    name=reg_name,
                    source=f"{run.info.artifact_uri}/weights",
                    run_id=run.info.run_id)
                print(f"registered model: {reg_name} v{mv.version}")
            except Exception as exc:
                print(f"WARNING: model registry unavailable: {exc}",
                      file=sys.stderr)
    finally:
        if mlflow is not None and mlflow.active_run() is not None:
            mlflow.end_run()

    print(f"best epoch {best['epoch']}: F1(damaged)={best['f1_damaged']:.4f} "
          f"acc={best['accuracy']:.4f}")
    print("classifier training completed.")
    return 0


if __name__ == "__main__":       # required on Windows (spawn-safe dataloaders)
    sys.exit(main())
