# Additional tables (audit round 2 — verified against args.yaml, logs, and PDF Chapter 3)

## T8-REVISED — Full proposal targets (Table 3.4) vs achieved, held-out test
**Replaces T8. Report per metric — including the honest near-miss.**

| metric | target (Table 3.4) | achieved — yolo26m·det11 | status |
|---|---|---|---|
| Detection mAP@0.5 | ≥ 0.75 | 0.753 | ✅ met (narrowly, +0.003) |
| Detection precision | ≥ 0.80 | 0.812 | ✅ met |
| Detection recall | ≥ 0.71 | **0.709** | ⚠️ **near-miss (−0.001)** |
| Detection F1 | ≥ 0.75 | 0.757 | ✅ met |
| Classifier precision | ≥ 0.85 | 0.938 | ✅ exceeded |
| Classifier recall | ≥ 0.76 | 0.952 | ✅ exceeded |
| Classifier F1 | ≥ 0.80 | 0.945 | ✅ exceeded |

Framing guidance: state plainly that recall lands 0.001 below target —
within single-run noise (one seed, no variance estimate) and within the
rounding granularity of the metric; note that yolo12m·det11 achieves
R = 0.725 (target met) at P = 0.751 (precision target missed), i.e. the
targets trade off against each other at this scale, and no single
configuration clears all four simultaneously. Honesty here pre-empts the
examiner's strongest attack; do NOT claim "all targets met".

## T10 — Training configuration (actual, from run args)

| setting | detection (all 8 runs) | classifier (all 3 runs) |
|---|---|---|
| framework | Ultralytics 8.4.115 / PyTorch | timm 1.0.28 / PyTorch |
| input size | 640 × 640 | 224 × 224 (crops, +10 % padding) |
| epochs | 200 (patience 100) | 30, best epoch kept by F1(damaged) |
| batch size | 16 | 32 |
| optimizer | auto → MuSGD (lr 0.01, momentum 0.9) | AdamW (lr 1e-4, wd 0.05) |
| LR schedule | lrf 0.01 (linear decay) | cosine annealing |
| loss | box + cls + DFL (7.5 / 0.5 / 1.5) | cross-entropy |
| augmentation | mosaic 1.0 (off last 10 ep), fliplr 0.5, scale ±0.5, translate 0.1, HSV (0.015/0.7/0.4), erasing 0.4, RandAugment | RandomResizedCrop (0.7–1.0), h-flip, ImageNet normalization |
| precision | AMP (mixed) | AMP (mixed) |
| seed | 42 (deterministic) | 42 |
| pretrained init | COCO weights (per model) | ImageNet weights (per backbone) |

Deviations from proposal §3.3.2/§3.4.4 to state: optimizer auto-resolved to
MuSGD rather than SGD/AdamW; 200 epochs (within the promised 100–300);
crop input 224 px rather than the proposed 256 px; BCE+class-weighting
replaced by plain CE (crop set nearly balanced: 4,815 / 5,449 — the
anticipated severe imbalance did not materialize, so Focal Loss was unnecessary).

## T11 — Positioning vs related work (Table 2.2 of the dissertation)

| work | task | metric | this project |
|---|---|---|---|
| Gao et al. [7], YOLOv11 | building cracks (single class) | mAP@0.5 = 0.886 | not comparable: 1 class vs 11 |
| **Li et al. [8], AHD-YOLO** | **conflict-zone damage (incl. Gaza imagery)** | **mAP = 0.705** | **0.753 — exceeds the closest comparator on a harder 11-class indoor task** |
| Yang et al. [9], YOLOv5l+ViT | bridge concrete damage | +0.081 mAP vs baseline | different domain |
| Ge et al. [10], DDNet | post-disaster buildings | F1 = 0.796 | detector F1 0.757 + classifier F1 0.945 (two-stage) |
| Liu et al. [11], YOLOv5s-GhostNet | prefabricated components | 0.99 crack acc. | single-defect task, not comparable |

Framing: AHD-YOLO is the only conflict-zone comparator — exceeding it by
+4.8 points mAP while detecting 11 element categories (vs damage regions)
from interior smartphone viewpoints is the headline claim; the others
differ in task/classes and belong in a "not directly comparable" caveat.

## T12 — Work plan (Table 3.7) vs actual

| phase (planned) | planned deliverable | actual |
|---|---|---|
| 1. Data collection & annotation (4 wk) | ~30,000 annotated images | 9,964 images / 14,702 objects, 4 annotators, sprint 2026-07-05→07-12; quality prioritized: validation gate + AI-assisted class review (see annotation_quality/) |
| 2. YOLO12 training (3 wk) | optimized detection model | **exceeded scope**: 8 configurations — YOLO12 *and* YOLO26 at n/s/m scales, two label schemes, 200 epochs each (2026-08-13/14, A100) |
| 3. Classifier development (2 wk) | damage classification model | 3 backbones trained + compared (DINOv2-S, EfficientNet-B0, Swin-Tiny); Swin selected |
| 4. Pipeline integration (2 wk) | working end-to-end system | full DVC pipeline (data→split→train→evaluate) + CI/CML automation; component-level test evaluation done; measured end-to-end chaining = future work |
| 5. Mobile deployment prep (2 wk) | mobile-ready packages | **deferred to Future Work** (design complete in §3.6; s-scale models identified as mobile fallback) |
| 6. Final evaluation & documentation (3 wk) | research report | held-out test protocol (evaluated once, 2026-08-16) + this dissertation |

Cite the proposal's own escape clause: timeline "subject to adjustment based
on data availability, computational resources, and experimental outcomes".

## T13 — Anticipated challenges (Table 3.8) vs what actually happened

| anticipated | materialized? | outcome |
|---|---|---|
| Scarce annotated damage imagery | YES | 9,964 collected vs 30k planned; augmentation + transfer learning used as proposed; targets still met |
| Class imbalance damaged/undamaged | NO (mild: 47/53) | plain CE sufficed; Focal Loss unnecessary |
| Subtle damage detection | PARTIALLY | binary damage worked (F1 0.945); *graded severity* remains future work |
| Limited mobile resources | DEFERRED | quantization/pruning not yet exercised (phase 5 deferred) |
| Visual similarity among classes | YES (different form) | resolved during *annotation*: 20→11 class consolidation after review; detector confusions remain for diffuse classes (Brick_Wall 0.623 AP50) |
| Lighting/camera inconsistency | YES | photometric augmentation (HSV/RandAugment) as proposed |
| Feedback-image management | N/A yet | DVC versioning infrastructure ready (as proposed) |

**Unanticipated engineering challenges actually encountered (lessons-learned material):**

| challenge | resolution |
|---|---|
| Google service accounts have zero storage quota on personal Drive — CI/server `dvc push` impossible by policy | pushes relayed through OAuth on a workstation; service account kept pull-only for CI |
| CUDA driver ↔ PyTorch build mismatch on cloud pod (cu130 wheel, CUDA-12.8 driver) silently trained on CPU | detected via GPU telemetry; pinned cu128 build; added a hard guard refusing full CPU training |
| Machine-specific absolute paths inside dataset config files broke portability across machines | runtime path rewriting in the training/eval scripts |
| Drive API throttling (SSL EOF) on bulk transfers | retry loops + GitHub Actions data cache (download once per dataset version) |
| Stale CI container (Python 3.8) incompatible with modern pins | plain runner + setup actions |
| CI smoke runs overwriting committed metrics before README regeneration | workspace restore step before publish |

## T14 — Model size & inference speed (measured)

| component | params | GFLOPs | weights | latency (RTX 3050 Ti laptop) |
|---|---|---|---|---|
| YOLO26m det11 (detector) | 20.4 M (fused) | 68.1 | 42 MB | 58 ms / image |
| Swin-Tiny (classifier) | 27.5 M | ~4.5 | 105 MB | 40 ms / crop |
| two-stage total (image with k elements) | — | — | 147 MB | ≈ 58 + 40k ms (batchable) |

Context for the deployment discussion: at the test-set average of ~1.5
objects/image the pipeline runs ≈ 118 ms/image on a mid-range laptop GPU —
mobile deployment (proposal §3.6) will require the promised quantization
and/or the s-scale detector (~½ the FLOPs at −1.5 pts val mAP50).

## T15 — Statistical context (single-seed caveat + intervals)

| quantity | value | 95 % CI (Wilson) |
|---|---|---|
| Swin test accuracy | 0.948 (n = 2,233) | [0.938, 0.957] |
| Swin damaged-recall | 0.952 (n = 1,047) | [0.938, 0.964] |
| Detection mAP@0.5 | 0.753 | single run, seed 42 — no variance estimate available |

State plainly: every configuration was trained once (seed 42) under a fixed
compute budget; the 0.753 vs 0.750 (yolo26m vs yolo12m) ordering and the
0.753-vs-0.75-target margin are within plausible single-run noise. The
deployment choice of yolo26m additionally rests on its decisive precision
advantage (+0.061), which is far outside noise. Air_Conditioner test
metrics rest on n = 12 instances — wide implicit interval, flag it.

## T16 — Generalization: validation vs held-out test (finalists)

| finalist | val | test | Δ |
|---|---|---|---|
| yolo26m·det11 mAP@0.5 | 0.738 | 0.753 | +0.015 |
| yolo12m·det11 mAP@0.5 | 0.750 | 0.750 | ±0.000 |
| yolo26n·det22 mAP@0.5 | 0.669 | 0.694 | +0.025 |
| swin accuracy | 0.958 | 0.948 | −0.010 |

Narrative: no overfitting signature (test ≥ val for detectors); the
held-out protocol corrected two val-based beliefs (m-scale ranking, Plan B
ceiling) — direct evidence the once-only test policy earned its keep.

## T17 — Plan B per-class test detail (yolo26n·det22, damaged vs intact)

Source: `tables/test_metrics_det_yolo26n_det22.json` (`per_class` holds all
22 entries). Report at least the damaged-class rows — damaged-element
recall is the operationally relevant number for Plan B — next to the
intact twins. Pair with the caveat that Plan A's "effective mAP50" product
(T7) is analytic, not a measured end-to-end chain; a measured chained
evaluation is listed in Future Work.
