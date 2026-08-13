# War Damage Assessment

War damage assessment via YOLO detection + damage classifiers, built on a
DVC + MLflow + CML MLOps stack.

## Latest Results

<!-- latest-results:start -->
_Generated: 2026-08-13T22:18_

| model | classes | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| yolo12n | 11 | 0.725 | 0.4738 | 0.7843 | 0.6892 |
| yolo12n | 22 | 0.6682 | 0.4322 | 0.7414 | 0.6171 |
| yolo26n | 11 | 0.7211 | 0.4671 | 0.7537 | 0.6843 |
| yolo26n | 22 | 0.6694 | 0.4353 | 0.7317 | 0.6331 |

> Note: variant-22 mAP is not directly comparable to variant-11 mAP (different class sets). It is included as reference only; the real Approach A vs B comparison happens after the classifier stage.

**Damage classifiers (Approach B stage 2 — damaged/undamaged):**

| model | backbone | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| dinov2 | vit_small_patch14_dinov2.lvd142m | 0.8717 | 0.8732 | 0.8522 | 0.8626 |
| efficientnet | efficientnet_b0 | 0.946 | 0.9565 | 0.928 | 0.942 |
| swin | swin_tiny_patch4_window7_224 | 0.9583 | 0.955 | 0.9568 | 0.9559 |

**Objects per split (11-class scheme):**

- train: 10264
- val: 2205
- test: 2233

<!-- latest-results:end -->
