# War Damage Assessment

War damage assessment via YOLO detection + damage classifiers, built on a
DVC + MLflow + CML MLOps stack.

## Latest Results

<!-- latest-results:start -->
_Generated: 2026-08-12T15:46_

| model | classes | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| yolo12n | 11 | 0.0016 | 0.0005 | 0.0057 | 0.1516 |
| yolo12n | 22 | 0.0 | 0.0 | 0.0 | 0.0 |
| yolo26n | 11 | 0.0015 | 0.0005 | 0.0022 | 0.1996 |
| yolo26n | 22 | 0.0 | 0.0 | 0.0 | 0.0 |

> Note: variant-22 mAP is not directly comparable to variant-11 mAP (different class sets). It is included as reference only; the real Approach A vs B comparison happens after the classifier stage.

**Objects per split (11-class scheme):**

- train: 10264
- val: 2205
- test: 2233

<!-- latest-results:end -->
