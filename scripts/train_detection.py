#!/usr/bin/env python3
"""Train one YOLO detection model on one dataset variant (11 or 22 classes).

Variant 22 (Approach A): damage-encoded classes, new_cls = cls + 11 * damage_flag.
Variant 11 (Approach B): base classes only, damage_flag stripped (a separate
classification stage decides damaged/undamaged per crop).

Config comes from params.yaml (train_detection: models, variants, epochs,
imgsz, batch, seed, smoke) so that `dvc repro` reacts to parameter changes.
CLI flags override params.yaml; --smoke forces smoke.epochs on a
smoke.fraction subset of the train split (CI/laptop plumbing check).

Usage (after prepare_split has built data/processed/detection{11,22}):
  python scripts/train_detection.py                          # first model, first variant
  python scripts/train_detection.py --model yolo26n --variant 22
  python scripts/train_detection.py --smoke                  # 1 epoch, 5% of train

Outputs:
  runs_detection/{model}_det{variant}/       weights, curves, confusion matrix
  reports/metrics_{model}_{variant}.json     summary metrics (DVC metrics file)
  reports/figures/{model}_{variant}/         training figures (also MLflow artifacts)
  mlflow.db                                  one run (experiment: war-damage-detection)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import os
from pathlib import Path

METRIC_KEYS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}

FIGURES = [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "val_batch0_pred.jpg",
]


def load_train_params(path: Path) -> dict:
    """train_detection section of params.yaml (single source of truth)."""
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))["train_detection"]


def main(argv: list[str] | None = None) -> int:
    # --params decides where the defaults come from, so parse it first
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--params", type=Path, default=Path("params.yaml"))
    pre_args, _ = pre.parse_known_args(argv)
    try:
        td = load_train_params(pre_args.params)
    except Exception as exc:
        print(f"cannot read train_detection from {pre_args.params}: {exc}",
              file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(
        description=__doc__, parents=[pre],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str,
                        choices=[str(m) for m in td["models"]],
                        default=str(td["models"][0]))
    parser.add_argument("--variant", type=int,
                        choices=[int(v) for v in td["variants"]],
                        default=int(td["variants"][0]))
    parser.add_argument("--smoke", action="store_true",
                        help="1 short run on a small train fraction (plumbing check)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override params.yaml (ignored in --smoke mode)")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args(argv)

    model, variant, smoke = args.model, args.variant, args.smoke
    if smoke:
        epochs = int(td["smoke"]["epochs"])
        fraction = float(td["smoke"]["fraction"])
    else:
        epochs = args.epochs if args.epochs is not None else int(td["epochs"])
        fraction = 1.0
    imgsz = args.imgsz if args.imgsz is not None else int(td["imgsz"])
    batch = args.batch if args.batch is not None else int(td["batch"])
    seed = int(td["seed"])

    data_yaml = Path(f"data/processed/detection{variant}/data.yaml")
    if not data_yaml.is_file():
        print(f"dataset file not found: {data_yaml}\n"
              "run the prepare_split stage first (dvc repro prepare_split)",
              file=sys.stderr)
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
        print("WARNING: mlflow not installed - training continues without tracking",
              file=sys.stderr)

    weights = f"{model}.pt"
    run_name = f"{model}-det{variant}" + ("-smoke" if smoke else "")
    num_classes = variant  # 11 = base scheme, 22 = damage-encoded scheme
    encoding_rule = ("new_cls = cls + 11 * damage_flag" if variant == 22
                     else "base 11 classes, damage_flag stripped")

    run = None
    if mlflow is not None:
        # steer the built-in ultralytics MLflow callback: it reuses our
        # already-active run for args + per-epoch metrics (desired) and
        # KEEP_RUN_ACTIVE stops it from ending the run before we append
        # summary metrics and artifacts below
        os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"
        # without this the callback set_experiments on trainer.args.project
        # (an absolute path), polluting the store with a junk experiment
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "war-damage-detection"
        uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        os.environ["MLFLOW_TRACKING_URI"] = uri  # keep the callback on the same store
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("war-damage-detection")
        run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags({
            "approach": "A" if variant == 22 else "B",
            "variant": str(variant),
            "model": model,
            "smoke": str(smoke),
        })
        # NOTE: values of keys the ultralytics callback re-logs (model, epochs,
        # imgsz, batch, seed) must match trainer.args exactly, otherwise the
        # callback hits a param-overwrite error and stops tracking the run
        mlflow.log_params({
            "model": weights,
            "variant": variant,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "seed": seed,
            "num_classes": num_classes,
            "encoding_rule": encoding_rule,
        })
        # self-description: record the exact class encoding with the run,
        # independent of params.yaml's future state
        classes_map = Path("reports/classes_map.json")
        if classes_map.is_file():
            mlflow.log_artifact(str(classes_map))
        else:
            print(f"WARNING: {classes_map} not found - encoding not attached to run",
                  file=sys.stderr)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: {device} | model: {weights} | variant: {variant} | "
          f"epochs: {epochs} | fraction: {fraction} | smoke: {smoke}")

    try:
        results = YOLO(weights).train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=seed,
            fraction=fraction,
            device=device,
            workers=2,
            # absolute path: ultralytics nests a relative project under
            # its settings runs_dir (runs/detect/...) instead of CWD
            project=str(Path("runs_detection").resolve()),
            name=f"{model}_det{variant}",
            exist_ok=True,
            verbose=True,
            plots=True,
        )

        md = getattr(results, "results_dict", {}) or {}
        summary = {k: round(float(md.get(v, -1.0)), 4)
                   for k, v in METRIC_KEYS.items()}
        print(f"=== RESULT {run_name} ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        if mlflow is not None and mlflow.active_run() is not None:
            native = {}
            for k, v in md.items():
                try:  # MLflow metric keys must not contain parentheses
                    native[k.replace("(", "").replace(")", "")] = float(v)
                except (TypeError, ValueError):
                    continue
            mlflow.log_metrics({**native, **summary})

        # summary metrics for DVC (dvc.yaml: metrics -> this file)
        metrics_path = Path(f"reports/metrics_{model}_{variant}.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"model": model, "variant": variant, "smoke": smoke,
                        "epochs": epochs, **summary}, indent=2) + "\n",
            encoding="utf-8")
        print(f"metrics written: {metrics_path}")

        run_dir = Path(getattr(results, "save_dir", "") or
                       Path("runs_detection") / f"{model}_det{variant}")

        fig_dir = Path("reports/figures") / f"{model}_{variant}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for fig in FIGURES:
            src = run_dir / fig
            if src.is_file():
                shutil.copy2(src, fig_dir / fig)
                copied += 1
        print(f"figures copied: {copied} -> {fig_dir}")
        if mlflow is not None and mlflow.active_run() is not None and copied:
            mlflow.log_artifacts(str(fig_dir), artifact_path="figures")

        best = run_dir / "weights" / "best.pt"
        if mlflow is not None and mlflow.active_run() is not None:
            if best.is_file():
                mlflow.log_artifact(str(best), artifact_path="weights")
                try:
                    # MLflow 3: runs:/ URIs resolve to logged-model entities,
                    # so register the raw artifact via create_model_version
                    reg_name = f"{model}-det{variant}"
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
            else:
                print(f"WARNING: {best} not found - weights not logged",
                      file=sys.stderr)
    finally:
        if mlflow is not None and mlflow.active_run() is not None:
            # end_run() defaults to FINISHED even on exceptions - mark
            # crashed runs FAILED so the experiment stays trustworthy
            mlflow.end_run("FAILED" if sys.exc_info()[0] is not None
                           else "FINISHED")

    return 0


if __name__ == "__main__":       # required on Windows (spawn-safe dataloaders)
    sys.exit(main())
