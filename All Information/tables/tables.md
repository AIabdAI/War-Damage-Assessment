# Paper-ready tables (all numbers verified against reports/*.json)

## T1 — Dataset composition (actual, final)

| property | value |
|---|---|
| Annotated image/label pairs | 9,964 |
| Annotated object instances | 14,702 |
| Classes (base scheme) | 11 (Beam removed during class review) |
| Split (image level, seed 42) | train 6,974 / val 1,495 / test 1,495 (70/15/15) |
| Objects per split | 10,264 / 2,205 / 2,233 |
| Damage-classification crops | 14,702 (same objects, 10 % padded) |
| Crop balance (train) | 4,815 damaged / 5,449 undamaged |
| Crop balance (test) | 1,047 damaged / 1,186 undamaged |

## T2 — Per-class object counts (train / val / test)

| class | train | val | test | total |
|---|---|---|---|---|
| Brick_Wall | 2,328 | 483 | 504 | 3,315 |
| Column | 1,854 | 411 | 425 | 2,690 |
| Light_Fixture | 1,416 | 292 | 298 | 2,006 |
| Door | 1,065 | 218 | 239 | 1,522 |
| Window | 970 | 200 | 213 | 1,383 |
| Floor_Tiles | 610 | 139 | 137 | 886 |
| Staircase | 608 | 117 | 124 | 849 |
| Sink | 464 | 114 | 93 | 671 |
| Wall_Cabinet | 476 | 91 | 101 | 668 |
| Toilet | 396 | 117 | 87 | 600 |
| Air_Conditioner | 77 | 23 | 12 | 112 |

## T3 — Detection results, VALIDATION split (200 epochs, A100-80GB)

| model · scheme | mAP50 | mAP50-95 | precision | recall | F1 | train time |
|---|---|---|---|---|---|---|
| yolo12m · det11 | **0.750** | 0.488 | 0.804 | **0.695** | 0.746 | 4 h 22 m |
| yolo26m · det11 | 0.738 | **0.490** | **0.814** | 0.693 | **0.748** | 3 h 58 m |
| yolo12s · det11 | 0.738 | 0.483 | 0.768 | 0.687 | 0.725 | 3 h 05 m |
| yolo26s · det11 | 0.733 | 0.486 | 0.801 | 0.685 | 0.738 | 3 h 06 m |
| yolo12n · det11 | 0.725 | 0.474 | 0.784 | 0.689 | 0.734 | 2 h 33 m |
| yolo26n · det11 | 0.721 | 0.467 | 0.754 | 0.684 | 0.717 | 2 h 40 m |
| yolo26n · det22 | 0.669 | 0.435 | 0.732 | 0.633 | 0.679 | 2 h 38 m |
| yolo12n · det22 | 0.668 | 0.432 | 0.741 | 0.617 | 0.674 | 2 h 32 m |

## T4 — Damage classifiers, VALIDATION split (30 epochs, 14,702 crops)

| model | backbone | F1 (damaged) | accuracy | best epoch | train time |
|---|---|---|---|---|---|
| swin | swin_tiny_patch4_window7_224 | **0.956** | **0.958** | 19 | 9 m 56 s |
| efficientnet | efficientnet_b0 | 0.942 | 0.946 | 25 | 9 m 36 s |
| dinov2 | vit_small_patch14_dinov2.lvd142m | 0.863 | 0.872 | 28 | 9 m 29 s |

## T5 — HELD-OUT TEST results (finalists, evaluated once)

| finalist | mAP50 | mAP50-95 | precision | recall | F1 |
|---|---|---|---|---|---|
| **yolo26m · det11** | **0.753** | **0.491** | **0.812** | 0.709 | **0.757** |
| yolo12m · det11 | 0.750 | 0.482 | 0.751 | **0.725** | 0.738 |
| yolo26n · det22 | 0.694 | 0.456 | 0.756 | 0.659 | 0.704 |

| classifier | crops | accuracy | precision (dmg) | recall (dmg) | F1 (dmg) |
|---|---|---|---|---|---|
| swin | 2,233 | 0.948 | 0.938 | 0.952 | 0.945 |

Swin test confusion: TP 997, FN 50, FP 66, TN 1,120.

## T6 — Per-class TEST AP50 (recommended detector, yolo26m · det11)

| class | AP50 | P | R |
|---|---|---|---|
| Staircase | 0.933 | 0.915 | 0.866 |
| Toilet | 0.915 | 0.879 | 0.836 |
| Light_Fixture | 0.888 | 0.913 | 0.848 |
| Door | 0.835 | 0.901 | 0.787 |
| Window | 0.823 | 0.846 | 0.770 |
| Sink | 0.684 | 0.807 | 0.630 |
| Column | 0.655 | 0.733 | 0.645 |
| Floor_Tiles | 0.647 | 0.711 | 0.647 |
| Wall_Cabinet | 0.643 | 0.719 | 0.594 |
| Air_Conditioner | 0.633 | 0.766 | 0.583 |
| Brick_Wall | 0.623 | 0.736 | 0.589 |

## T7 — Approach comparison (Plan A two-stage vs Plan B single-stage), TEST

| quantity | value |
|---|---|
| Plan A effective mAP50 (yolo26m-det11 × swin acc) | 0.753 × 0.948 = **0.714** |
| Plan B measured mAP50 (yolo26n-det22) | **0.694** |
| Margin | +0.020 (+2.8 % relative) |
| Break-even classifier accuracy | 0.694 / 0.753 = 92.2 % |
| Swin margin over break-even | 94.8 % − 92.2 % = +2.6 points |
| det22 scheme penalty (nano, val) | −0.052…−0.057 mAP50 (−7.2…−7.8 % rel.) |

## T8 — Proposal targets vs achieved

| target (Chapter 3.5.6) | achieved (held-out test) | status |
|---|---|---|
| Detection mAP@0.5 ≥ 0.75 | 0.753 (yolo26m · det11) | ✅ met |
| Classification F1 ≥ 0.75 | 0.945 (swin, damaged class) | ✅ exceeded (+0.195) |

## T9 — Compute summary

| phase | runs | hardware | wall time |
|---|---|---|---|
| Detection training | 8 × 200 epochs | A100-SXM4-80GB | ≈ 24.6 h total |
| Classifier training | 3 × 30 epochs | A100-SXM4-80GB | ≈ 29 m total |
| Held-out test evaluation | 3 det + 1 cls | RTX 3050 Ti (laptop) | ≈ 8 m |
