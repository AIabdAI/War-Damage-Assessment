# Model Compression for On-Device Deployment — Results Report

Compression of the trained models for mobile inference (dissertation §3.6),
covering the post-training techniques of §3.6.4 and the distribution
infrastructure the future Flutter application will use.

- **Date:** 2026-09-05
- **Source models:** unchanged. Every artefact here is a *new* file under
  `models_mobile/`; `runs_detection/` and `runs_classification/` were read
  only.
- **Pipeline:** `compression/dvc.yaml` (`dvc repro --single-item`)
- **Tracking:** MLflow experiment `war-damage-compression`, one run per
  model × variant × split, each registered as `{model}-{variant}-mobile`
- **Artefacts:** 16 ONNX models (4 models × 4 variants), DVC-tracked and
  pushed to the Drive remote

## 1. Compression achieved (size)

| model | source `.pt` | ONNX FP32 | FP16 | INT8 dynamic | INT8 static |
|---|---|---|---|---|---|
| yolo26m_det11 (detector) | 42.0 MB | 77.96 MB | 39.05 (2.0×) | **19.96 (3.9×)** | 20.39 (3.8×) |
| yolo12s_det11 (compact detector) | 19.0 MB | 35.55 MB | 17.86 (2.0×) | **9.38 (3.8×)** | 9.83 (3.6×) |
| swin (classifier) | 105.0 MB | 107.63 MB | 54.36 (2.0×) | 29.44 (3.7×) | **29.15 (3.7×)** |
| efficientnet (compact classifier) | 15.6 MB | 15.27 MB | 7.68 (2.0×) | 4.11 (3.7×) | 4.59 (3.3×) |

INT8 delivers the ~4× reduction §3.6.4 predicts; FP16 delivers exactly 2×.
Size alone, however, does not decide deployability — §2 shows why.

## 2. Accuracy after compression — the decisive result

### 2.1 Detector (yolo26m_det11)

Measured on a fixed, deterministic 300-image subset of each split (seed 42).
Every variant sees the *same* images, so the deltas between them are exact;
absolute values differ slightly from the full-split figures reported in the
main results chapter (full-split PyTorch test mAP@0.5 = 0.7527).

| variant | val mAP50 | test mAP50 | test mAP50-95 | test P | test R | size |
|---|---|---|---|---|---|---|
| FP32 ONNX (reference) | 0.6677 | **0.7569** | 0.4833 | 0.7924 | 0.6943 | 77.96 MB |
| **INT8 dynamic** | 0.6404 | **0.7102** | 0.4471 | 0.7373 | 0.6942 | **19.96 MB** |
| INT8 static | 0.0000 | **0.0000** | 0.0 | 0.0 | 0.0 | 20.39 MB |

- The FP32 ONNX export is faithful to PyTorch (0.757 vs 0.753 full-split) —
  the export path itself loses nothing.
- **INT8 dynamic costs −0.047 mAP@0.5 (−6.2 % relative) for 3.9× compression.**
- **INT8 static fails completely: the model emits no detection above the
  confidence threshold at all.** Verified directly: on three test images the
  FP32 model returns detections at confidence 0.784/0.905 and the
  dynamically-quantised model returns the same detections at 0.740/0.907,
  while the statically-quantised model returns none.

### 2.2 Damage classifiers (full splits, 2,205 val / 2,233 test crops)

| model | variant | val acc | test acc | test F1 (damaged) | size |
|---|---|---|---|---|---|
| **swin** | FP32 ONNX | 0.9596 | 0.9498 | 0.9469 | 107.63 MB |
| **swin** | **INT8 static** | 0.9478 | **0.9454** | 0.9419 | **29.15 MB** |
| efficientnet | FP32 ONNX | 0.9438 | 0.9413 | — | 15.27 MB |
| efficientnet | FP16 | 0.9438 | — | — | 7.68 MB |
| efficientnet | INT8 static | 0.7388 | 0.7734 | — | 4.59 MB |
| efficientnet | INT8 dynamic | 0.5442 | — | — | 4.11 MB |

- **Swin-Tiny quantises almost losslessly: −0.44 accuracy points on the
  held-out test set for 3.7× compression.**
- **EfficientNet-B0 collapses under INT8** — −16.8 points static, and dynamic
  quantisation drops it to 0.544, near the 53 % majority-class baseline
  (i.e. the model stops discriminating).
- FP16 is exactly lossless (0.9438 = FP32) but only halves the size.

## 3. The central finding: quantisation sensitivity is architecture-specific,
and it inverts between the two stages

| model | INT8 **dynamic** | INT8 **static** |
|---|---|---|
| YOLO26m detector | ✅ works (−6.2 % rel.) | ❌ total failure (0 detections) |
| Swin-Tiny classifier | ⚠️ pathologically slow on CPU | ✅ works (−0.4 pts) |
| EfficientNet-B0 classifier | ❌ collapses to 0.544 | ❌ collapses to 0.773 |

Interpretation:

- **The detector needs *dynamic* quantisation.** Static quantisation fixes
  activation ranges from calibration data; YOLO26's end-to-end head carries
  box coordinates in pixel scale alongside logits, so a single per-tensor
  range cannot cover both and the detection head is destroyed. Dynamic
  quantisation derives ranges per inference and survives.
- **Swin needs *static* quantisation** — its transformer activations are
  well-behaved, and the dynamic path pays a large runtime cost re-ranging
  every MatMul.
- **EfficientNet-B0 cannot be quantised at all** by post-training methods.
  Its depthwise-separable convolutions have per-channel ranges that
  per-tensor activation quantisation cannot represent, a known failure mode
  for this family. Quantisation-Aware Training (§3.6.4) would be required.

**This inverts the intuitive choice.** The "mobile-friendly" compact CNN is
the one that cannot be deployed quantised, while the heavier transformer
compresses almost losslessly. Choosing a mobile model on parameter count
alone would have produced an unusable system.

## 4. Recommended deployment bundle

| role | model | variant | size | measured quality |
|---|---|---|---|---|
| detection | yolo26m_det11 | INT8 dynamic | 19.96 MB | test mAP@0.5 0.710 (−6.2 % vs FP32) |
| damage classification | swin | INT8 static | 29.15 MB | test accuracy 0.945 (−0.4 pts vs FP32) |
| **total on-device footprint** | | | **≈ 49 MB** | vs 186 MB unquantised (3.8× smaller) |

Fallback if a device or runtime rejects INT8: the FP16 pair (39.05 + 54.36 =
93 MB), which is numerically near-lossless.

A second deployment advantage favours the recommended detector:
**YOLO26m exports with NMS inside the ONNX graph** (`[x1, y1, x2, y2, conf,
class]`, 300 rows), so the Flutter client filters by confidence and uses the
boxes directly. YOLO12s emits raw anchors `(1, 15, 8400)` and would require
NMS to be implemented in Dart.

## 5. Inference latency — not reported

Latency was measured on the development laptop and the results were not
reproducible enough to publish: the same model measured 430 ms and 4733 ms
in consecutive runs, and yolo26m (68 GFLOPs) measured *faster* than
yolo12s (21 GFLOPs), which is physically implausible. The cause is
contention and thermal behaviour on a Windows laptop, not the models.

x86 laptop timings would not transfer to ARM mobile SoCs in any case.
**On-device benchmarking on target Android/iOS hardware is required** and is
listed as future work. What can be stated from these runs is the ordering
observed consistently: INT8 dynamic detector inference was faster than FP32,
and FP16 on CPU was slower than FP32 (CPU has no native FP16 path — FP16
targets GPU/NNAPI delegates).

## 6. Distribution infrastructure (verified end-to-end)

`serve/model_api.py` serves the artefacts to the future Flutter application.
All endpoints were tested live against the built registry:

| check | result |
|---|---|
| `GET /health` | `{"status":"ok","registry_present":true}` |
| `GET /api/v1/version` (cheap update check) | release 1.0.0 |
| `GET /api/v1/bundle/latest` | detector + classifier, with sizes, SHA-256, metrics, preprocessing and output format |
| `GET /api/v1/models/{id}/download` | 29.15 MB transferred; SHA-256 **matches the registry** |
| re-download with `If-None-Match` | **HTTP 304, 0 bytes** — the device never re-downloads |
| `GET /api/v1/models?production=true` | the four production artefacts |

The registry declares per-model preprocessing and output format, so the app
hard-codes nothing; the production variant is set per architecture
(`--production-detector int8_dynamic`, `--production-classifier int8_static`)
following the measurements above.

## 7. Engineering issues encountered and resolved

| issue | resolution |
|---|---|
| Generic FP16 conversion (`onnxconverter_common`) produced **unloadable** YOLO graphs — Resize nodes emitting float32 into float16 consumers | use ultralytics' native `half=True` export; added a load-check so an invalid artefact can never be published |
| Ultralytics fed CUDA tensors to a CPU ONNX session ("no data transfer registered") | pass `device="cpu"` for ONNX validation |
| Partial re-runs overwrote manifests/results instead of merging, silently dropping variants | merge-on-write in both scripts |
| ONNX evaluation preprocessing squashed images to square while training resized the shorter side | aspect-preserving resize in both the evaluator and the INT8 calibration reader |

## 8. Limitations and next steps

1. Detector accuracy is measured on 300-image subsets (variant *deltas* are
   exact; absolute values come from the full-split evaluation in the main
   results chapter). A full-split rerun of the chosen variant is cheap on a
   GPU machine and should precede release.
2. TFLite and Core ML conversion (§3.6.3) are not yet produced; the ONNX
   artefacts are the source format for both.
3. Quantisation-Aware Training (§3.6.4) is the untried technique that would
   likely rescue EfficientNet-B0 and could recover the detector's 6 %.
4. Pruning and knowledge distillation (§3.6.4) remain unexplored.
5. On-device latency, memory and battery measurements on real Android/iOS
   hardware are required before a deployment decision is final.

---
*Sources: `models_mobile/*/*/compression_manifest.json`,
`reports/compression/*.json`, MLflow `war-damage-compression`,
`serve/model_registry.json`. Reproduce with
`cd compression && dvc repro --single-item`.*
