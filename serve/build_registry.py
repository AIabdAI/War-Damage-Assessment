#!/usr/bin/env python3
"""Build the model-distribution registry consumed by the download API.

Collects every compressed artefact under models_mobile/, attaches its
measured accuracy/latency (reports/compression/*.json), computes a SHA-256
for on-device integrity verification, and writes serve/model_registry.json.

The registry is the contract between the backend and the future Flutter
application: it declares, per model, the version, checksum, size, format,
runtime, and the exact preprocessing the device must apply.

Usage:
  python serve/build_registry.py                 # all variants
  python serve/build_registry.py --production int8_static
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_VERSION = "1.0"
# semantic version of the model artefacts themselves; bump when models are
# retrained or recompressed so devices can detect a newer release
MODEL_RELEASE = "1.0.0"

PREPROCESSING = {
    "detector": {
        "input_size": [640, 640],
        "layout": "NCHW",
        "color_order": "RGB",
        "scale": "divide_by_255",
        "normalization": "none",
        "letterbox": True,
        "conf_threshold": 0.25,
        "iou_threshold": 0.7,
        "max_detections": 300,
    },
    "classifier": {
        "input_size": [224, 224],
        "layout": "NCHW",
        "color_order": "RGB",
        "scale": "divide_by_255",
        "normalization": {"mean": [0.485, 0.456, 0.406],
                          "std": [0.229, 0.224, 0.225]},
        "resize_then_center_crop": [256, 224],
        "classes": ["damaged", "undamaged"],
    },
}

CLASS_NAMES = ["Brick_Wall", "Column", "Staircase", "Floor_Tiles", "Sink",
               "Wall_Cabinet", "Window", "Door", "Air_Conditioner",
               "Light_Fixture", "Toilet"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_format(onnx_path: Path, kind: str) -> dict:
    """Describe the model's output so the device needs no hard-coded assumptions."""
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        shape = [d.dim_value or d.dim_param
                 for d in m.graph.output[0].type.tensor_type.shape.dim]
    except Exception:
        return {}
    if kind == "classifier":
        return {"shape": shape, "layout": "logits[batch, 2]",
                "classes": ["damaged", "undamaged"], "postprocess": "argmax"}
    if len(shape) == 3 and shape[-1] == 6:
        return {"shape": shape, "layout": "[x1, y1, x2, y2, confidence, class_id]",
                "nms_included": True,
                "postprocess": "filter by confidence threshold; boxes ready to use"}
    return {"shape": shape, "layout": "[batch, 4 + num_classes, anchors]",
            "nms_included": False,
            "postprocess": "transpose, decode boxes, then apply NMS on device"}


def load_metrics(kind: str, name: str) -> dict:
    """Measured metrics per variant, keyed by split."""
    out: dict[str, dict] = {}
    for split in ("val", "test"):
        p = Path("reports/compression") / f"{kind}_{name}_{split}.json"
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            for vname, entry in data.get("variants", {}).items():
                out.setdefault(vname, {})[split] = {
                    k: v for k, v in entry.items()
                    if k in ("mAP50", "mAP50_95", "precision", "recall", "f1",
                             "accuracy", "precision_damaged", "recall_damaged",
                             "f1_damaged", "cpu_latency_ms")
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--production", default="int8_static",
                    help="variant flagged as the production default for mobile")
    ap.add_argument("--out", type=Path, default=Path("serve/model_registry.json"))
    args = ap.parse_args()

    registry = {
        "registry_version": REGISTRY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": MODEL_RELEASE,
        "detection_classes": CLASS_NAMES,
        "models": [],
    }

    for kind in ("detector", "classifier"):
        root = Path("models_mobile") / kind
        if not root.is_dir():
            continue
        for model_dir in sorted(root.iterdir()):
            man = model_dir / "compression_manifest.json"
            if not man.is_file():
                continue
            manifest = json.loads(man.read_text(encoding="utf-8"))
            name = manifest["name"]
            metrics = load_metrics(kind, name)
            for vname, info in manifest["variants"].items():
                f = model_dir / info["file"]
                if not f.is_file():
                    continue
                model_id = f"{name}-{vname}"
                registry["models"].append({
                    "model_id": model_id,
                    "kind": kind,
                    "base_model": name,
                    "variant": vname,
                    "version": MODEL_RELEASE,
                    "format": "onnx",
                    "opset": 13,
                    "runtime": "onnxruntime-mobile",
                    "compression_technique": info.get("technique"),
                    "quantization_type": info.get("quant_type", "none"),
                    "size_bytes": f.stat().st_size,
                    "size_mb": info["size_mb"],
                    "compression_ratio_vs_fp32": info.get("compression_ratio_vs_fp32"),
                    "sha256": sha256(f),
                    "path": str(f).replace("\\", "/"),
                    "download_url": f"/api/v1/models/{model_id}/download",
                    "preprocessing": PREPROCESSING[kind],
                    "output_format": output_format(f, kind),
                    "metrics": metrics.get(vname, {}),
                    "production": vname == args.production,
                })

    # the deployment bundle the Flutter app asks for in one call
    def pick(kind: str) -> str | None:
        cands = [m for m in registry["models"]
                 if m["kind"] == kind and m["production"]]
        if not cands:
            cands = [m for m in registry["models"] if m["kind"] == kind]
        if not cands:
            return None
        preferred = {"detector": "yolo26m_det11", "classifier": "swin"}[kind]
        for m in cands:
            if m["base_model"] == preferred:
                return m["model_id"]
        return cands[0]["model_id"]

    registry["bundle"] = {
        "bundle_version": MODEL_RELEASE,
        "detector": pick("detector"),
        "classifier": pick("classifier"),
        "pipeline": ["capture_image", "record_gps", "preprocess", "detect",
                     "crop_elements", "classify_damage", "store_local", "sync"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"registry written: {args.out} ({len(registry['models'])} artefacts)")
    print(f"bundle: detector={registry['bundle']['detector']} "
          f"classifier={registry['bundle']['classifier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
