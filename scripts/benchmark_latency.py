#!/usr/bin/env python3
"""Measure clean single-sample CPU latency for every compressed variant.

Accuracy evaluation and latency measurement are separated on purpose: the
accuracy pass is long and can share the machine, but latency is only
meaningful on an otherwise idle CPU. Run this last, with nothing else going.

Two thread settings are reported:
  threads=4   mobile-representative (mid-range phone big-core count)
  threads=all upper bound on this workstation

Usage:
  python scripts/benchmark_latency.py                 # all variants
  python scripts/benchmark_latency.py --runs 50

Outputs:
  reports/compression/latency_benchmark.json
  (also merged into reports/compression/<kind>_<name>_<split>.json)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

VARIANT_ORDER = ["fp32", "fp16", "int8_dynamic", "int8_static"]


def measure(onnx_path: Path, shape: tuple[int, ...], threads: int,
            runs: int) -> float:
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    x = np.random.rand(*shape).astype(dtype)
    for _ in range(5):                       # warm-up
        sess.run(None, {inp.name: x})
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {"median_ms": round(samples[len(samples) // 2], 1),
            "p90_ms": round(samples[int(len(samples) * 0.9)], 1),
            "min_ms": round(samples[0], 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    out = {"machine": platform.processor() or platform.machine(),
           "note": "single sample, batch=1, ONNX Runtime CPU provider",
           "models": {}}
    for kind in ("detector", "classifier"):
        shape = (1, 3, 640, 640) if kind == "detector" else (1, 3, 224, 224)
        root = Path("models_mobile") / kind
        if not root.is_dir():
            continue
        for model_dir in sorted(root.iterdir()):
            for vname in VARIANT_ORDER:
                f = model_dir / f"{vname}.onnx"
                if not f.is_file():
                    continue
                key = f"{model_dir.name}-{vname}"
                print(f"benchmarking {key} ...", flush=True)
                try:
                    out["models"][key] = {
                        "kind": kind,
                        "size_mb": round(f.stat().st_size / 1048576, 2),
                        "threads_4": measure(f, shape, 4, args.runs),
                        "threads_all": measure(f, shape, 0, args.runs),
                    }
                except Exception as exc:   # an invalid graph must not stop the run
                    print(f"   FAILED: {str(exc)[:120]}")
                    out["models"][key] = {"kind": kind, "error": str(exc)[:200]}
                    continue
                m = out["models"][key]
                print(f"   4-thread median {m['threads_4']['median_ms']} ms | "
                      f"all-core median {m['threads_all']['median_ms']} ms")

    dst = Path("reports/compression/latency_benchmark.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {dst}")

    # merge the clean numbers into the per-split result files
    for res in sorted(Path("reports/compression").glob("*_*.json")):
        if res.name == "latency_benchmark.json":
            continue
        data = json.loads(res.read_text(encoding="utf-8"))
        changed = False
        for vname, entry in data.get("variants", {}).items():
            key = f"{data['name']}-{vname}"
            if key in out["models"]:
                entry["cpu_latency_ms"] = out["models"][key]["threads_4"]["median_ms"]
                entry["cpu_latency_all_cores_ms"] = \
                    out["models"][key]["threads_all"]["median_ms"]
                changed = True
        if changed:
            res.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"updated: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
