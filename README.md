# War Damage Assessment

War damage assessment via YOLO detection + damage classifiers, built on a
DVC + MLflow + CML MLOps stack.

## Latest Results

<!-- latest-results:start -->
_Generated: 2026-08-16T12:30_

### Validation results (val split, model selection)

| model | classes | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| yolo12m | 11 | 0.7496 | 0.488 | 0.8042 | 0.6951 |
| yolo12n | 11 | 0.725 | 0.4738 | 0.7843 | 0.6892 |
| yolo12n | 22 | 0.6682 | 0.4322 | 0.7414 | 0.6171 |
| yolo12s | 11 | 0.738 | 0.4829 | 0.7677 | 0.687 |
| yolo26m | 11 | 0.7384 | 0.4899 | 0.8135 | 0.6927 |
| yolo26n | 11 | 0.7211 | 0.4671 | 0.7537 | 0.6843 |
| yolo26n | 22 | 0.6694 | 0.4353 | 0.7317 | 0.6331 |
| yolo26s | 11 | 0.7334 | 0.4856 | 0.8007 | 0.6848 |

> Note: variant-22 mAP is not directly comparable to variant-11 mAP (different class sets). It is included as reference only; the real Approach A vs B comparison happens after the classifier stage.

**Damage classifiers (Approach B stage 2 — damaged/undamaged):**

| model | backbone | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| dinov2 | vit_small_patch14_dinov2.lvd142m | 0.8717 | 0.8732 | 0.8522 | 0.8626 |
| efficientnet | efficientnet_b0 | 0.946 | 0.9565 | 0.928 | 0.942 |
| swin | swin_tiny_patch4_window7_224 | 0.9583 | 0.955 | 0.9568 | 0.9559 |

### Held-out test results (finalists, evaluated once)

| model | mAP50 | mAP50-95 | precision | recall | F1 |
|---|---|---|---|---|---|
| yolo26m_det11 | 0.7527 | 0.4912 | 0.8115 | 0.7086 | 0.7566 |
| yolo12m_det11 | 0.7498 | 0.4816 | 0.7512 | 0.7245 | 0.7376 |
| yolo26n_det22 | 0.6941 | 0.4562 | 0.756 | 0.659 | 0.7042 |

| classifier | crops | accuracy | precision (damaged) | recall (damaged) | F1 (damaged) |
|---|---|---|---|---|---|
| swin | 2233 | 0.9481 | 0.9379 | 0.9522 | 0.945 |

> Full analysis and per-class breakdown: [reports/comparison_report.md](reports/comparison_report.md)

**Objects per split (11-class scheme):**

- train: 10264
- val: 2205
- test: 2233

<!-- latest-results:end -->
