# War Damage Assessment — Approach Comparison Report

**Status: FINAL** — covers every completed run of the experiment matrix:
six detection models on the base scheme, two on the damage-encoded scheme,
and three damage classifiers. Nano models and classifiers trained 2026-08-13
(Pod 1); small/medium models trained 2026-08-14 (Pod 2).

- **Dataset:** 9,964 image/label pairs, 14,702 annotated objects, 11 classes,
  deterministic 70/15/15 split (seed 42) — see `reports/split_manifest.json`.
- **Hardware:** NVIDIA A100-SXM4-80GB. Detection: 200 epochs each.
  Classifiers: 30 epochs each on 14,702 padded crops.
- **Tracking:** MLflow experiments `war-damage-detection` and
  `war-damage-classification` (databases versioned as `mlflow_pod1.db.dvc` /
  `mlflow_pod2.db.dvc`); pipeline fingerprint in `dvc.lock`.

## The two competing designs

| | Plan A — two-stage | Plan B — single-stage |
|---|---|---|
| Detection | 11 base classes (det11) | 22 classes = 11 × {intact, damaged} (det22, `new_cls = cls + 11 · damage_flag`) |
| Damage decision | dedicated crop classifier | folded into the detector |
| Components to deploy | 2 models | 1 model |

## Detection results (200 epochs, full data)

| model · scheme | mAP50 | mAP50-95 | precision | recall | F1¹ | train time² |
|---|---|---|---|---|---|---|
| **yolo12m · det11** | **0.750** | 0.488 | 0.804 | **0.695** | 0.746 | 4 h 22 m |
| yolo26m · det11 | 0.738 | **0.490** | **0.814** | 0.693 | **0.748** | 3 h 58 m |
| yolo12s · det11 | 0.738 | 0.483 | 0.768 | 0.687 | 0.725 | 3 h 05 m |
| yolo26s · det11 | 0.733 | 0.486 | 0.801 | 0.685 | 0.738 | 3 h 06 m |
| yolo12n · det11 | 0.725 | 0.474 | 0.784 | 0.689 | 0.734 | 2 h 33 m |
| yolo26n · det11 | 0.721 | 0.467 | 0.754 | 0.684 | 0.717 | 2 h 40 m |
| yolo26n · det22 | 0.669 | 0.435 | 0.732 | 0.633 | 0.679 | 2 h 38 m |
| yolo12n · det22 | 0.668 | 0.432 | 0.741 | 0.617 | 0.674 | 2 h 32 m |

¹ F1 derived as 2·P·R / (P+R) from final validation precision/recall.
² Ultralytics cumulative training clock (`results.csv`); MLflow wall-clock
runs 1–3 % higher (setup + validation + artifact logging).

### What scale bought

- **yolo12 leads on mAP50 at every scale** (n 0.725 → s 0.738 → m 0.750);
  each size step adds roughly +0.012 mAP50 for ~21–42 % more training time
  per step.
- **yolo26 catches up with size**: at nano it trails yolo12 on every det11
  metric, but
  **yolo26m takes best mAP50-95 (0.490), best precision (0.814), and best
  F1 (0.748)** — the architecture pays off at medium scale, favoring
  localization quality and fewer false positives over raw mAP50.
- The s-models are the efficiency sweet spot: ~98 % of medium-scale mAP50
  for roughly three-quarters of the training time (72–78 %).
- det22 was trained at nano scale only (see caveats): both nano runs show a
  consistent scheme penalty of −0.052 to −0.057 mAP50 (−7.2 % to −7.8 %
  relative) versus their det11 twins.

## Classifier results (30 epochs, damaged/undamaged crops — Pod 1)

| model | backbone | F1 (damaged) | accuracy | best epoch | train time³ |
|---|---|---|---|---|---|
| **swin** | swin_tiny_patch4_window7_224 | **0.956** | **0.958** | 19 | 9 m 56 s |
| efficientnet | efficientnet_b0 | 0.942 | 0.946 | 25 | 9 m 36 s |
| dinov2 | vit_small_patch14_dinov2.lvd142m | 0.863 | 0.872 | 28 | 9 m 29 s |

³ Wall-clock durations from MLflow run records.

Swin-Tiny leads on F1, accuracy, and damaged-class recall; EfficientNet-B0
is marginally ahead on damaged-class precision (0.957 vs 0.955). DINOv2's
frozen-feature pedigree did not translate into an advantage under identical
fine-tuning conditions.

## Central analysis — Plan A vs Plan B (final)

Plan A's end-to-end quality is approximately the det11 detector's quality
degraded by the classifier's error rate (every detected box is re-judged by
the classifier):

- **Plan A (yolo12m-det11 + swin):** effective mAP50 ≈ 0.750 × 0.958 =
  **0.718**; effective recall ≈ 0.695 × 0.958 = **0.666**.
- **Plan B (best measured, yolo26n-det22):** mAP50 = **0.669**,
  recall = 0.633.

**Break-even:** Plan A beats Plan B whenever classifier accuracy exceeds the
det22/det11 quality ratio. Against the strongest det11 detector that
threshold is 0.669 / 0.750 = **89.3 %** (computed on unrounded metrics) —
Swin's measured 95.8 % clears it
by 6.5 points. The margin *grew* with detector scale: +0.025 effective mAP50
at nano (interim report) → **+0.049 (+7.3 % relative) at medium**.

**Final verdict: Plan A — yolo12m-det11 detection + Swin-Tiny damage
classification.** It wins on every effective-quality measure, and its
advantage widened as detectors improved, because better detection compounds
through the classifier while the single-stage scheme pays its class-splitting
penalty at any scale.

Plan B retains its deployment virtues — one model, one inference pass,
simpler failure analysis — and remains a reasonable choice where
integration simplicity outweighs ~5 points of effective mAP50.

Caveats, stated honestly:
- det22 was never trained at s/m scale. Extrapolating yolo12's measured nano
  penalty (−7.8 % relative) onto yolo12m suggests a hypothetical
  yolo12m-det22 near **0.69 mAP50** — still below Plan A's 0.718, but this
  is an estimate, not a measurement.
- The effective-quality product treats classifier errors as uniform across
  classes and ignores localization interplay; det22 aggregate recall is not
  its damaged-classes-only recall. A per-class error analysis on the held-out
  test split is the recommended next step before deployment.

## Figures

### Recommended pipeline — yolo12m · det11 (best mAP50)
![training curves](figures/yolo12m_11/results.png)
![confusion matrix](figures/yolo12m_11/confusion_matrix_normalized.png)
![PR curve](figures/yolo12m_11/BoxPR_curve.png)
![validation predictions](figures/yolo12m_11/val_batch0_pred.jpg)

### Runner-up — yolo26m · det11 (best mAP50-95, precision, F1)
![training curves](figures/yolo26m_11/results.png)
![confusion matrix](figures/yolo26m_11/confusion_matrix_normalized.png)

### Best single-stage — yolo26n · det22
![training curves](figures/yolo26n_22/results.png)
![confusion matrix](figures/yolo26n_22/confusion_matrix_normalized.png)

### Recommended classifier — swin
![training curves](figures/cls_swin/results.png)
![confusion matrix](figures/cls_swin/confusion_matrix.png)

### Full figure index (all runs)

| run | curves | confusion | PR curve | predictions |
|---|---|---|---|---|
| yolo12m·11 | [results](figures/yolo12m_11/results.png) | [matrix](figures/yolo12m_11/confusion_matrix_normalized.png) | [PR](figures/yolo12m_11/BoxPR_curve.png) | [val](figures/yolo12m_11/val_batch0_pred.jpg) |
| yolo26m·11 | [results](figures/yolo26m_11/results.png) | [matrix](figures/yolo26m_11/confusion_matrix_normalized.png) | [PR](figures/yolo26m_11/BoxPR_curve.png) | [val](figures/yolo26m_11/val_batch0_pred.jpg) |
| yolo12s·11 | [results](figures/yolo12s_11/results.png) | [matrix](figures/yolo12s_11/confusion_matrix_normalized.png) | [PR](figures/yolo12s_11/BoxPR_curve.png) | [val](figures/yolo12s_11/val_batch0_pred.jpg) |
| yolo26s·11 | [results](figures/yolo26s_11/results.png) | [matrix](figures/yolo26s_11/confusion_matrix_normalized.png) | [PR](figures/yolo26s_11/BoxPR_curve.png) | [val](figures/yolo26s_11/val_batch0_pred.jpg) |
| yolo12n·11 | [results](figures/yolo12n_11/results.png) | [matrix](figures/yolo12n_11/confusion_matrix_normalized.png) | [PR](figures/yolo12n_11/BoxPR_curve.png) | [val](figures/yolo12n_11/val_batch0_pred.jpg) |
| yolo26n·11 | [results](figures/yolo26n_11/results.png) | [matrix](figures/yolo26n_11/confusion_matrix_normalized.png) | [PR](figures/yolo26n_11/BoxPR_curve.png) | [val](figures/yolo26n_11/val_batch0_pred.jpg) |
| yolo12n·22 | [results](figures/yolo12n_22/results.png) | [matrix](figures/yolo12n_22/confusion_matrix_normalized.png) | [PR](figures/yolo12n_22/BoxPR_curve.png) | [val](figures/yolo12n_22/val_batch0_pred.jpg) |
| yolo26n·22 | [results](figures/yolo26n_22/results.png) | [matrix](figures/yolo26n_22/confusion_matrix_normalized.png) | [PR](figures/yolo26n_22/BoxPR_curve.png) | [val](figures/yolo26n_22/val_batch0_pred.jpg) |
| cls swin | [results](figures/cls_swin/results.png) | [matrix](figures/cls_swin/confusion_matrix.png) | — | — |
| cls efficientnet | [results](figures/cls_efficientnet/results.png) | [matrix](figures/cls_efficientnet/confusion_matrix.png) | — | — |
| cls dinov2 | [results](figures/cls_dinov2/results.png) | [matrix](figures/cls_dinov2/confusion_matrix.png) | — | — |

## Conclusion & recommendation

Deploy **Plan A: yolo12m-det11 + Swin-Tiny** (effective mAP50 ≈ 0.718,
effective recall ≈ 0.666). Where inference cost matters more than the last
point of quality, **yolo12s-det11 + Swin** delivers ~98 % of that at a
fraction of the compute; where false positives are the dominant concern,
**yolo26m-det11** is the better detector half. Recommended follow-ups:
per-class test-split error analysis, and — only if Plan B must be revisited —
a single yolo12m-det22 run to replace the extrapolated caveat with a
measurement.

Note on provenance: Pod 2 also re-ran the three classifiers as part of its
pipeline; those near-duplicate runs were deliberately not merged — the
published classifier results and registered models remain Pod 1's canonical
runs.

---
*Generated from: `reports/metrics_*.json` (8 detection + 3 classifier runs),
`runs_detection/*/results.csv`, MLflow run records (`mlflow_pod1.db`,
`mlflow_pod2.db`). Weights and databases versioned in DVC; run fingerprints
in `dvc.lock`.*
