#!/usr/bin/env python3
"""Evaluate compressed (ONNX) models and register them in MLflow.

For every variant produced by compress_models.py this measures accuracy on
the requested split(s) and single-image CPU latency (the mobile-representative
path), then logs one MLflow run per variant carrying the full deployment
record required for model management:

  original model name + version, compression technique, quantisation type,
  original/compressed size, metrics before and after compression, inference
  time, target framework, model format and opset.

Selection policy: compression variants are CHOSEN on the validation split;
the test split is reported for the record (the models themselves are frozen,
so this does not leak into training decisions).

Usage:
  python scripts/evaluate_compressed.py --kind classifier --name swin --splits val,test
  python scripts/evaluate_compressed.py --kind detector --name yolo26m_det11 --splits test
  python scripts/evaluate_compressed.py --all --splits val,test

Outputs:
  reports/compression/<kind>_<name>_<split>.json
  MLflow experiment "war-damage-compression" (+ registered models)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EXPERIMENT = "war-damage-compression"
TARGET_FRAMEWORK = "ONNX Runtime Mobile (Android/iOS); source format for TFLite + Core ML conversion"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VARIANT_ORDER = ["fp32", "fp16", "int8_dynamic", "int8_static"]

DETECTORS = ["yolo26m_det11", "yolo12s_det11"]
CLASSIFIERS = ["swin", "efficientnet"]


def baseline_metrics(kind: str, name: str, split: str) -> dict:
    """Original (PyTorch) metrics already recorded by the training/eval stages."""
    reports = Path("reports")
    if kind == "detector":
        if split == "test":
            p = reports / f"test_metrics_det_{name}.json"
            keys = ("mAP50", "mAP50_95", "precision", "recall")
        else:
            model, variant = name.split("_det")
            p = reports / f"metrics_{model}_{variant}.json"
            keys = ("mAP50", "mAP50_95", "precision", "recall")
    else:
        p = (reports / f"test_metrics_cls_{name}.json" if split == "test"
             else reports / f"metrics_cls_{name}.json")
        keys = ("accuracy", "precision_damaged", "recall_damaged", "f1_damaged")
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: data[k] for k in keys if k in data}


def time_cpu(onnx_path: Path, shape: tuple[int, ...], runs: int = 30) -> float:
    """Single-sample CPU latency in ms (mobile-representative)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4          # comparable to a mid-range phone
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    itype = sess.get_inputs()[0].type
    dtype = np.float16 if "float16" in itype else np.float32
    x = np.random.rand(*shape).astype(dtype)
    for _ in range(3):
        sess.run(None, {iname: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {iname: x})
    return round((time.perf_counter() - t0) / runs * 1000, 1)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def build_subset(dataset_dir: Path, split: str, n: int, seed: int = 42) -> Path:
    """Deterministic N-image subset (hardlinked) for fast variant comparison.

    Every variant is scored on the SAME images, so the deltas between them -
    which is what a compression study needs - stay exact. Absolute values come
    from the full-split evaluations reported in the main results chapter.
    """
    import os
    import random
    import shutil
    dst = Path("runs_compression") / f"subset_{dataset_dir.name}_{split}_{n}"
    if (dst / "images").is_dir() and any((dst / "images").iterdir()):
        return dst
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    files = sorted((dataset_dir / split / "images").glob("*"))
    rng = random.Random(seed)
    rng.shuffle(files)
    for f in files[:n]:
        for src, out in ((f, dst / "images" / f.name),
                         (dataset_dir / split / "labels" / f"{f.stem}.txt",
                          dst / "labels" / f"{f.stem}.txt")):
            if not src.is_file():
                continue
            try:
                os.link(src, out)
            except OSError:
                shutil.copy2(src, out)
    return dst


def eval_detector(onnx_path: Path, name: str, split: str,
                  max_images: int = 0) -> dict:
    import yaml
    from ultralytics import YOLO
    variant = 22 if name.endswith("det22") else 11
    dataset_dir = Path(f"data/processed/detection{variant}")
    cfg = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    if max_images:
        sub = build_subset(dataset_dir, split, max_images)
        cfg["path"] = sub.resolve().as_posix()
        cfg["train"] = cfg["val"] = cfg["test"] = "images"
        split = "val"                    # the subset is exposed as the val key
        data_yaml = Path("runs_detection") / f"data_{variant}_subset.local.yaml"
    else:
        cfg["path"] = dataset_dir.resolve().as_posix()
        data_yaml = Path("runs_detection") / f"data_{variant}.local.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    res = YOLO(str(onnx_path), task="detect").val(
        data=str(data_yaml), split=split, plots=False, verbose=False,
        # the ONNX session runs on the CPU provider; without this ultralytics
        # feeds CUDA tensors and ORT fails with "no data transfer registered"
        device="cpu",
        project=str(Path("runs_compression").resolve()),
        name=f"{name}_{onnx_path.stem}_{split}", exist_ok=True)
    md = getattr(res, "results_dict", {}) or {}
    out = {
        "mAP50": round(float(md.get("metrics/mAP50(B)", -1.0)), 4),
        "mAP50_95": round(float(md.get("metrics/mAP50-95(B)", -1.0)), 4),
        "precision": round(float(md.get("metrics/precision(B)", -1.0)), 4),
        "recall": round(float(md.get("metrics/recall(B)", -1.0)), 4),
    }
    p, r = out["precision"], out["recall"]
    out["f1"] = round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0
    return out


def eval_classifier(onnx_path: Path, split: str, batch: int = 32) -> dict:
    import onnxruntime as ort
    from PIL import Image
    root = Path("data/processed/classification") / split
    classes = ["damaged", "undamaged"]
    files = [(p, ci) for ci, c in enumerate(classes)
             for p in sorted((root / c).glob("*.jpg"))]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    imgsz = 224
    resize = int(round(imgsz / 0.875))

    def prep(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("RGB")
            # torchvision Resize(int) scales the SHORTER side and keeps aspect
            w, h = im.size
            scale = resize / min(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.BILINEAR)
            w, h = im.size
            left, top = (w - imgsz) // 2, (h - imgsz) // 2
            im = im.crop((left, top, left + imgsz, top + imgsz))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)

    tp = fp = fn = tn = 0
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        x = np.stack([prep(p) for p, _ in chunk]).astype(dtype)
        logits = sess.run(None, {iname: x})[0]
        preds = np.asarray(logits).argmax(1)
        for (_, truth), pred in zip(chunk, preds):
            if pred == 0 and truth == 0:
                tp += 1
            elif pred == 0 and truth == 1:
                fp += 1
            elif pred == 1 and truth == 0:
                fn += 1
            else:
                tn += 1
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"crops": total,
            "accuracy": round((tp + tn) / total, 4) if total else 0.0,
            "precision_damaged": round(precision, 4),
            "recall_damaged": round(recall, 4),
            "f1_damaged": round(f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["detector", "classifier"])
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--variants", default=",".join(VARIANT_ORDER))
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--max-images", type=int, default=0,
                    help="detectors: score a deterministic N-image "
                         "subset (0 = full split)")
    args = ap.parse_args(argv)

    if args.all:
        jobs = [("detector", n) for n in DETECTORS] + \
               [("classifier", n) for n in CLASSIFIERS]
    elif args.kind and args.name:
        jobs = [(args.kind, args.name)]
    else:
        ap.error("use --all, or both --kind and --name")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    want = [v.strip() for v in args.variants.split(",") if v.strip()]

    mlflow = None
    if not args.no_mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                                   "sqlite:///mlflow.db"))
            mlflow.set_experiment(EXPERIMENT)
        except ImportError:
            print("mlflow unavailable - results still written to reports/",
                  file=sys.stderr)

    out_dir = Path("reports/compression")
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, name in jobs:
        model_dir = Path("models_mobile") / kind / name
        man_path = model_dir / "compression_manifest.json"
        if not man_path.is_file():
            print(f"SKIP {kind}/{name}: run compress_models.py first", file=sys.stderr)
            continue
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        shape = (1, 3, 640, 640) if kind == "detector" else (1, 3, 224, 224)

        for split in splits:
            base = baseline_metrics(kind, name, split)
            prev_path = out_dir / f"{kind}_{name}_{split}.json"
            prev_variants = {}
            if prev_path.is_file():      # keep variants from earlier partial runs
                try:
                    prev_variants = json.loads(
                        prev_path.read_text(encoding="utf-8")).get("variants", {})
                except Exception:
                    pass
            results = {"kind": kind, "name": name, "split": split,
                       "baseline_pytorch": base, "variants": dict(prev_variants)}
            for vname in VARIANT_ORDER:
                if vname not in want or vname not in manifest["variants"]:
                    continue
                info = manifest["variants"][vname]
                onnx_path = model_dir / info["file"]
                print(f"\n--- {kind}/{name} [{vname}] {split} ---", flush=True)
                t0 = time.perf_counter()
                try:
                    metrics = (eval_detector(onnx_path, name, split,
                                             args.max_images)
                               if kind == "detector"
                               else eval_classifier(onnx_path, split))
                except Exception as exc:
                    print(f"  eval FAILED: {exc}", file=sys.stderr)
                    results["variants"][vname] = {"error": str(exc), **info}
                    continue
                latency = time_cpu(onnx_path, shape)
                entry = {**info, **metrics, "cpu_latency_ms": latency,
                         "eval_seconds": round(time.perf_counter() - t0, 1)}
                # accuracy delta vs the FP32 ONNX baseline of the same split
                key = "mAP50" if kind == "detector" else "accuracy"
                if "fp32" in results["variants"] and key in results["variants"]["fp32"]:
                    entry[f"delta_{key}_vs_fp32"] = round(
                        entry[key] - results["variants"]["fp32"][key], 4)
                results["variants"][vname] = entry
                print(f"  {key}={entry.get(key)}  size={info['size_mb']}MB  "
                      f"cpu={latency}ms", flush=True)

                if mlflow is not None:
                    with mlflow.start_run(run_name=f"{name}-{vname}-{split}"):
                        mlflow.set_tags({
                            "stage": "compression",
                            "kind": kind,
                            "original_model": name,
                            "original_model_version": "v1",
                            "compression_technique": info.get("technique", vname),
                            "quantization_type": info.get("quant_type", "none"),
                            "calibration": info.get("calibration", "n/a"),
                            "target_framework": TARGET_FRAMEWORK,
                            "model_format": "ONNX",
                            "onnx_opset": "13",
                            "split": split,
                        })
                        mlflow.log_params({
                            "variant": vname,
                            "original_size_mb": manifest["source_checkpoint"]["size_mb"],
                            "onnx_fp32_size_mb": manifest["variants"]["fp32"]["size_mb"],
                            "compressed_size_mb": info["size_mb"],
                            "compression_ratio_vs_fp32": info.get(
                                "compression_ratio_vs_fp32"),
                            "input_shape": str(shape),
                        })
                        loggable = {k: v for k, v in entry.items()
                                    if isinstance(v, (int, float))}
                        mlflow.log_metrics(loggable)
                        for bk, bv in base.items():
                            mlflow.log_metric(f"baseline_{bk}", bv)
                        # log the binary once per (model, variant), not per split
                        first_split = split == splits[0]
                        if first_split:
                            mlflow.log_artifact(str(onnx_path), artifact_path="model")
                        try:
                            if not first_split:
                                raise RuntimeError("registered with the first split")
                            client = mlflow.MlflowClient()
                            reg = f"{name}-{vname}-mobile"
                            try:
                                client.create_registered_model(reg)
                            except Exception:
                                pass
                            run = mlflow.active_run()
                            client.create_model_version(
                                name=reg,
                                source=f"{run.info.artifact_uri}/model",
                                run_id=run.info.run_id)
                        except Exception as exc:
                            print(f"  registry warning: {exc}", file=sys.stderr)

            dst = out_dir / f"{kind}_{name}_{split}.json"
            dst.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(f"written: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
