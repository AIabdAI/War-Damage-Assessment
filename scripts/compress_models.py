#!/usr/bin/env python3
"""Compress trained models for on-device deployment (dissertation section 3.6).

Produces mobile-ready variants of the EXISTING models without touching them:
the originals under runs_detection/ and runs_classification/ are read-only
inputs; every artefact is written to models_mobile/.

Techniques (section 3.6.4), all post-training so no retraining is needed:
  fp32          ONNX baseline export (reference for size/accuracy deltas)
  fp16          half-precision weights (~2x smaller, GPU/NNAPI friendly)
  int8_dynamic  weights quantised to INT8, activations quantised at runtime
  int8_static   full INT8 via calibration on TRAIN images (never test data)

Usage:
  python scripts/compress_models.py --kind detector   --name yolo26m_det11
  python scripts/compress_models.py --kind classifier --name swin
  python scripts/compress_models.py --all

Outputs:
  models_mobile/<kind>/<name>/<technique>.onnx
  models_mobile/<kind>/<name>/compression_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CALIB_SAMPLES = 128          # calibration images, drawn from the TRAIN split
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DETECTORS = ["yolo26m_det11", "yolo12s_det11"]
CLASSIFIERS = ["swin", "efficientnet"]


def mb(path: Path) -> float:
    return round(path.stat().st_size / 1048576, 2)


# --------------------------------------------------------------------------- #
# calibration readers (static INT8)
# --------------------------------------------------------------------------- #
class DetectorCalibReader:
    """Feeds resized TRAIN images to the static quantiser."""

    def __init__(self, images: list[Path], input_name: str, imgsz: int = 640):
        from PIL import Image
        self._data = []
        for p in images:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((imgsz, imgsz))
                arr = np.asarray(im, dtype=np.float32) / 255.0
            self._data.append({input_name: arr.transpose(2, 0, 1)[None]})
        self._it = iter(self._data)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._data)


class ClassifierCalibReader:
    """Feeds preprocessed TRAIN crops (same transform as evaluation)."""

    def __init__(self, images: list[Path], input_name: str, imgsz: int = 224):
        from PIL import Image
        resize = int(round(imgsz / 0.875))
        self._data = []
        for p in images:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size                      # aspect-preserving, as in eval
                scale = resize / min(w, h)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.BILINEAR)
                w, h = im.size
                left, top = (w - imgsz) // 2, (h - imgsz) // 2
                im = im.crop((left, top, left + imgsz, top + imgsz))
                arr = np.asarray(im, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            self._data.append(
                {input_name: arr.transpose(2, 0, 1)[None].astype(np.float32)})
        self._it = iter(self._data)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._data)


def sample_train_images(root: Path, n: int, seed: int = 42) -> list[Path]:
    import random
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_detector(name: str, out_dir: Path, imgsz: int = 640,
                    half: bool = False) -> Path:
    """Export a detector to ONNX.

    NOTE: FP16 must come from ultralytics' native half export (needs a GPU).
    The generic onnxconverter_common FP16 pass produces an UNLOADABLE graph
    for YOLO models - it leaves Resize nodes emitting float32 into float16
    consumers ("Type Error: ... does not match expected type (tensor(float16))").
    """
    from ultralytics import YOLO
    weights = Path("runs_detection") / name / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")
    kw = {}
    if half:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 export needs CUDA; skip fp16 on this machine")
        kw = {"half": True, "device": 0}
    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True,
                                 dynamic=False, opset=13, verbose=False, **kw))
    dst = out_dir / ("fp16.onnx" if half else "fp32.onnx")
    shutil.move(str(exported), dst)          # the original .pt stays untouched
    return dst


def export_detector_fp32(name: str, out_dir: Path, imgsz: int = 640) -> Path:
    return export_detector(name, out_dir, imgsz, half=False)


def export_classifier_fp32(name: str, out_dir: Path) -> Path:
    import timm
    import torch
    ckpt_path = Path("runs_classification") / name / "weights" / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone, imgsz = ckpt["backbone"], int(ckpt["imgsz"])
    kw = {"pretrained": False, "num_classes": 2}
    if "patch14" in backbone:
        kw["img_size"] = imgsz
    model = timm.create_model(backbone, **kw)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dst = out_dir / "fp32.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, imgsz, imgsz), str(dst),
        input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, do_constant_folding=True,
    )
    # carry the deployment contract inside the model file
    import onnx
    m = onnx.load(str(dst))
    for k, v in {"backbone": backbone, "imgsz": str(imgsz),
                 "classes": "damaged,undamaged",
                 "normalization": "imagenet"}.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(dst))
    return dst


def to_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(src))
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), str(dst))
    return dst


def preserve_metadata(src: Path, dst: Path) -> None:
    """ORT quantisation drops embedded metadata; restore it."""
    import onnx
    s, d = onnx.load(str(src)), onnx.load(str(dst))
    have = {e.key for e in d.metadata_props}
    added = False
    for e in s.metadata_props:
        if e.key not in have:
            n = d.metadata_props.add()
            n.key, n.value = e.key, e.value
            added = True
    if added:
        onnx.save(d, str(dst))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["detector", "classifier"])
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--techniques", default="fp32,fp16,int8_dynamic,int8_static")
    args = ap.parse_args(argv)

    if args.all:
        jobs = [("detector", n) for n in DETECTORS] + \
               [("classifier", n) for n in CLASSIFIERS]
    elif args.kind and args.name:
        jobs = [(args.kind, args.name)]
    else:
        ap.error("use --all, or both --kind and --name")

    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_dynamic,
                                          quantize_static)

    for kind, name in jobs:
        out_dir = Path("models_mobile") / kind / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {kind}: {name} ===")
        man_path = out_dir / "compression_manifest.json"
        manifest = {"kind": kind, "name": name, "variants": {}}
        if man_path.is_file():      # keep variants from earlier runs
            try:
                prev = json.loads(man_path.read_text(encoding="utf-8"))
                manifest["variants"].update(prev.get("variants", {}))
            except Exception:
                pass

        t0 = time.perf_counter()
        fp32 = (export_detector_fp32(name, out_dir) if kind == "detector"
                else export_classifier_fp32(name, out_dir))
        manifest["variants"]["fp32"] = {
            "file": fp32.name, "size_mb": mb(fp32),
            "technique": "ONNX export (FP32 baseline)",
            "export_s": round(time.perf_counter() - t0, 1)}
        print(f"  fp32          {mb(fp32):7.2f} MB")

        sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        del sess

        if "fp16" in techniques:
            dst = out_dir / "fp16.onnx"
            try:
                if kind == "detector":
                    # native half export - the generic converter breaks YOLO graphs
                    export_detector(name, out_dir, half=True)
                else:
                    to_fp16(fp32, dst)
                    preserve_metadata(fp32, dst)
                import onnxruntime as _ort         # fail fast on an invalid graph
                _ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
                manifest["variants"]["fp16"] = {
                    "file": dst.name, "size_mb": mb(dst),
                    "technique": "post-training FP16 weight conversion"}
                print(f"  fp16          {mb(dst):7.2f} MB")
            except Exception as exc:
                print(f"  fp16 FAILED: {exc}", file=sys.stderr)

        if "int8_dynamic" in techniques:
            dst = out_dir / "int8_dynamic.onnx"
            try:
                quantize_dynamic(str(fp32), str(dst), weight_type=QuantType.QUInt8)
                preserve_metadata(fp32, dst)
                manifest["variants"]["int8_dynamic"] = {
                    "file": dst.name, "size_mb": mb(dst),
                    "technique": "post-training dynamic quantisation",
                    "quant_type": "QUInt8 weights, activations ranged at runtime"}
                print(f"  int8_dynamic  {mb(dst):7.2f} MB")
            except Exception as exc:
                print(f"  int8_dynamic FAILED: {exc}", file=sys.stderr)

        if "int8_static" in techniques:
            dst = out_dir / "int8_static.onnx"
            try:
                if kind == "detector":
                    imgs = sample_train_images(
                        Path("data/processed/detection11/train/images"), CALIB_SAMPLES)
                    reader = DetectorCalibReader(imgs, input_name)
                else:
                    imgs = sample_train_images(
                        Path("data/processed/classification/train"), CALIB_SAMPLES)
                    reader = ClassifierCalibReader(imgs, input_name)
                quantize_static(
                    str(fp32), str(dst), reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax,
                )
                preserve_metadata(fp32, dst)
                manifest["variants"]["int8_static"] = {
                    "file": dst.name, "size_mb": mb(dst),
                    "technique": "post-training static quantisation (QDQ)",
                    "quant_type": "QUInt8 activations / QInt8 per-channel weights",
                    "calibration": f"{len(imgs)} TRAIN images, MinMax"}
                print(f"  int8_static   {mb(dst):7.2f} MB")
            except Exception as exc:
                print(f"  int8_static FAILED: {exc}", file=sys.stderr)

        base = manifest["variants"]["fp32"]["size_mb"]
        for v in manifest["variants"].values():
            v["compression_ratio_vs_fp32"] = round(base / v["size_mb"], 2)
        src_pt = (Path("runs_detection") / name / "weights" / "best.pt"
                  if kind == "detector"
                  else Path("runs_classification") / name / "weights" / "best.pt")
        manifest["source_checkpoint"] = {"path": str(src_pt).replace("\\", "/"),
                                         "size_mb": mb(src_pt)}
        manifest["variants"] = {k: manifest["variants"][k]
                                for k in ("fp32", "fp16", "int8_dynamic",
                                          "int8_static")
                                if k in manifest["variants"]}
        man_path.write_text(json.dumps(manifest, indent=2) + "\n",
                            encoding="utf-8")
        n_variants = len(manifest["variants"])
        print(f"  manifest -> {man_path} ({n_variants} variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
