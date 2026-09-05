# Capstone completion package — master index

Materials to complete the three remaining chapters of
`capstone_project_final.pdf` (Results and Experimental Validation,
Conclusions, Future Work). Every number here is verified against the
project's tracked artifacts (`reports/*.json`, `dvc.lock`, MLflow).

**Package contents**

| folder | contents |
|---|---|
| `tables/` | `tables.md` (all 9 tables) + `tables.tex` (LaTeX-ready) + the 4 raw test JSONs |
| `figures/` | 16 paper-ready images (naming: `fig_<what>_<model>.png`) |
| `diagrams/` | 4 Mermaid workflow diagrams + full `dvc dag` output |
| `snippets/` | `dvc_pipeline.yaml`, `evaluation_pipeline.yaml`, `params.yaml`, `cml_workflow.yml` |
| `comparison_report.md` | the full internal results report (source of the analysis) |

---

## ⚠️ Proposal-vs-actual reconciliation (address early in the Results chapter)

The proposal (Chapters 1–3) predates the experiments. State these deltas
explicitly — examiners respect documented scope evolution:

| proposal said | actually done | why |
|---|---|---|
| 20 indoor element categories | **11 classes** | class review merged/removed unreliable categories (incl. Beam, removed after annotation review); ids remapped, gate-validated |
| ~30,000 annotated images | **9,964 images / 14,702 objects** | realistic collection capacity under war-time constraints; quality (validation gate, class review tooling) prioritized over raw count |
| YOLO12 as the detector | **YOLO12 and YOLO26 both evaluated at 3 scales** | stronger experiment: 8 detector configurations; YOLO26m won on test |
| DINOv2 / EfficientNet-B3 / MobileNetV4 classifier candidates | **DINOv2-S / EfficientNet-B0 / Swin-Tiny evaluated** | B0 fits edge-deployment budget; Swin-Tiny added and won decisively |
| single two-stage design | **two competing designs compared** (Plan A two-stage vs Plan B single-stage det22) | turns the design choice into a measured research question — the core experimental contribution |

Targets (full Table 3.4 set — see **T8-REVISED** in `tables_additional.md`):
detection mAP 0.753 ≥ 0.75 ✅ (narrow), precision 0.812 ≥ 0.80 ✅,
**recall 0.709 vs ≥ 0.71 — near-miss by 0.001, report honestly**,
F1 0.757 ≥ 0.75 ✅; all three classifier targets exceeded (P 0.938 ≥ 0.85,
R 0.952 ≥ 0.76, F1 0.945 ≥ 0.80). Do NOT claim "all targets met".

---

## Chapter: Results and Experimental Validation

Suggested structure with materials mapped:

**x.1 Experimental setup**
- Hardware: NVIDIA A100-SXM4-80GB (cloud pods) for training; detection 200
  epochs @ 640 px, batch 16, seed 42; classifiers 30 epochs @ 224 px,
  batch 32, AdamW lr 1e-4. Test evaluation on RTX 3050 Ti laptop.
- Dataset: Table **T1** (composition) + **T2** (per-class counts).
  Deterministic 70/15/15 image-level split, seed 42, versioned split
  manifest (leak-free by construction; crops inherit the image's split).
- Reproducibility: DVC pipeline (snippet `dvc_pipeline.yaml`, diagram
  **D2/D4**), every run fingerprinted in `dvc.lock`, MLflow experiments
  (`war-damage-detection`, `war-damage-classification`,
  `war-damage-test-eval`), weights + MLflow DBs versioned on the DVC remote.
- Compute totals: Table **T9**.

**x.2 Detection results (validation)**
- Table **T3** (all 8 configurations). Figures: `fig_det_curves_yolo26m.png`,
  `fig_det_curves_yolo12m.png`, `fig_det_cm_val_yolo26m.png`,
  `fig_det_pr_val_yolo26m.png`, `fig_det_predictions_val.png`.
- Narrative points: YOLO12 leads mAP50 at every scale on val
  (0.725→0.738→0.750, ≈+0.012 per size step for 21–42 % more train time);
  YOLO26 favors precision/localization and takes mAP50-95 + precision + F1
  at medium scale; s-scale = efficiency sweet spot (~98 % of m-scale mAP50
  at ~72–78 % of the time).
- det22 scheme penalty: −0.052…−0.057 mAP50 (−7.2…−7.8 % relative) —
  halving per-class examples while doubling classes measurably hurts
  (figures `fig_det22_curves_yolo26n.png`, `fig_det22_cm_val_yolo26n.png`).

**x.3 Damage classification results (validation)**
- Table **T4**. Figures: `fig_cls_curves_swin.png`, `fig_cls_cm_val_swin.png`
  (+ efficientnet/dinov2 curves for comparison).
- Swin-Tiny best on F1/accuracy/damaged-recall; EfficientNet-B0 marginally
  better damaged-precision (0.957 vs 0.955); DINOv2 clearly behind under
  identical fine-tuning (0.863 F1).

**x.4 Held-out test validation** ← the chapter's centerpiece
- Protocol: test split (1,495 images / 2,233 objects+crops) never touched
  during training or model selection; finalists evaluated exactly once via
  a separate DVC pipeline (snippet `evaluation_pipeline.yaml`) — protects
  the set's statistical validity.
- Tables **T5** (test results), **T6** (per-class). Figures:
  `fig_test_cm_yolo26m.png`, `fig_test_pr_yolo26m.png`,
  `fig_test_predictions_yolo26m.png`, `fig_test_cm_swin.png`,
  `fig_test_cm_yolo26n22.png`.
- Key findings to report honestly:
  1. Test flips the m-scale ranking: YOLO26m 0.753 vs YOLO12m 0.750 mAP50
     (within noise) but +0.061 precision and +0.019 F1 → YOLO26m is the
     deployment pick.
  2. Plan B generalized better than validation implied (0.694 test vs
     0.669 val) while Swin dipped slightly (0.948 vs 0.958) — evidence the
     held-out protocol caught val-selection optimism.
  3. Per-class: distinctive shapes excel (Staircase 0.933, Toilet 0.915,
     Light_Fixture 0.888); diffuse/rare classes lag (Brick_Wall 0.623,
     Air_Conditioner 0.633 — only 12 test instances, wide CI).

**x.5 Approach comparison (the research question)**
- Diagram **D3**; Table **T7**.
- Plan A effective test mAP50 = 0.753 × 0.948 = **0.714** vs Plan B
  **0.694** → +0.020 (+2.8 %). Break-even classifier accuracy 92.2 %;
  Swin clears by 2.6 points.
- Verdict: **Plan A (YOLO26m-det11 + Swin-Tiny)** on quality; Plan B
  remains attractive for single-model deployment simplicity — state the
  margin honestly and the caveat that det22 was not trained at m-scale
  (extrapolated ≈0.69, below Plan A).
- Targets table **T8-REVISED** (in `tables_additional.md`) closes the
  chapter: 6 of 7 targets met/exceeded; detection recall 0.709 vs ≥ 0.71 is
  an honest near-miss (pair with T15's single-run caveat).

**x.6 MLOps validation (optional but strong)**
- The pipeline itself was validated: CI (GitHub Actions + CML) smoke-runs
  the full pipeline per push and publishes metric tables + figures as
  commit comments; README auto-refreshes from tracked metrics (snippet
  `cml_workflow.yml`, diagram **D2**). Data + 11 training runs + 4 test
  evaluations reproducible from `dvc.lock`.

## Chapter: Conclusions

Draft skeleton (all claims backed by the tables):
1. Built and validated an end-to-end damage-assessment pipeline for Gaza's
   residential reconstruction context: 9,964-image / 11-class dataset with
   validated annotations; 8 detector + 3 classifier configurations trained;
   design question (two-stage vs single-stage) answered empirically on a
   once-only held-out test set.
2. Six of the seven Table 3.4 targets met or exceeded on the held-out
   test split (detection mAP 0.753, P 0.812, F1 0.757; classifier P/R/F1
   0.938/0.952/0.945); detection recall 0.709 misses its 0.71 target by
   0.001 — within single-run noise (state plainly, with T15's caveat).
3. Recommended system: YOLO26m det11 detector + Swin-Tiny classifier —
   effective test mAP@0.5 ≈ 0.714, exceeding the best single-stage
   alternative by 2.8 % relative; per-class analysis identifies where it
   is deployment-ready (fixtures ≥0.82 AP50) vs where caution is needed
   (walls/cabinets ~0.62–0.65).
4. Methodological contribution: full MLOps discipline (DVC-versioned data
   and models, MLflow tracking, CI/CML reporting, isolated test-evaluation
   pipeline) — every reported number is reproducible from a git commit.
5. Honest limitations: dataset 1/3 of proposed size and 11 (not 20)
   classes; single-region imagery; test set evaluated once but only 1,495
   images (Air_Conditioner n=12); damage is binary, not graded; edge
   deployment designed but not yet implemented (proposal Chapter 3.6).

## Chapter: Future Work

Ordered by evidence-backed priority:
1. **Targeted data collection** for the weak classes — Brick_Wall,
   Wall_Cabinet, Air_Conditioner (T6 shows exactly where more data moves
   the needle most).
2. **Graded damage severity** (e.g. minor/moderate/severe/destroyed)
   instead of binary — the classifier headroom (0.945 F1) suggests
   capacity for finer labels.
3. **Edge deployment** (proposal 3.6): export YOLO26m + Swin-Tiny to
   TFLite/ONNX, quantize, measure on-device latency/accuracy; the s-scale
   detectors (0.733–0.738 val mAP50 at ~½ the compute) are the fallback if
   m-scale exceeds the mobile budget.
4. **A measured m-scale det22 run** to replace the extrapolated Plan B
   ceiling with a measurement (single training run).
5. **Mobile app + feedback loop** (proposal 3.7): GPS-tagged capture,
   user-correction harvesting, periodic retraining through the existing
   DVC/MLflow pipeline; spatial damage maps in the data warehouse.
6. **Robustness studies**: cross-region generalization, low-light/blur
   sensitivity (proposal Table 3.5), calibration of confidence thresholds
   per class.
7. **Scaling the registry to deployment**: model registry versions exist
   (yolo26m-det11 v1, swin-cls v1) — wire a serving/packaging step.

---

*Numbers cross-checked against: `reports/test_metrics_*.json`,
`reports/metrics_*.json`, `reports/comparison_report.md` (independently
fact-checked twice), `dvc.lock`, MLflow experiments. GitHub:
`github.com/AIabdAI/War-Damage-Assessment` @ `e5ac3957`.*


---

## Round 2 additions (completeness audit)

| addition | purpose |
|---|---|
| `tables/tables_additional.md` | **T8-REVISED** (full per-metric targets incl. the recall near-miss), T10 training config (actual hyperparameters + augmentation — a guaranteed viva question), T11 related-work positioning (AHD-YOLO 0.705 is the key comparator), T12 work-plan actual-vs-planned, T13 challenges retrospective + real engineering lessons-learned, T14 model size/speed (20.4M/68.1 GFLOPs/42MB/58ms + Swin 27.5M/105MB/40ms), T15 statistical caveats (Wilson CIs, single-seed, Air_Conditioner n=12), T16 val-vs-test generalization, T17 Plan B per-class pointer |
| `annotation_quality/` | dataset-credibility dossier: failing→passing gate reports, 1,490-box human review + 706 corrections, YOLO-World AI-assisted QA (446 flags), provenance census (65% team photos / 5% web), annotator process |
| 4 new figures + `figures/CAPTIONS.md` | test ground-truth mosaic (pairs with predictions), dataset-statistics plot, augmentation batch example, test F1-vs-confidence curve; captions manifest prevents val/test mislabeling |
| `facts_extra.json` | provenance, timeline milestones, Wilson CIs, measured size/latency — machine-readable |
| `comparison_report.md` (updated) | val-era "Central analysis" retitled to *validation estimates* with a pointer to the test section — internal contradiction removed |

Known remaining gaps (deliberately not produced — candidates for the viva
"future work" answer): measured end-to-end Plan A chain (the effective-mAP50
product is analytic), per-element-class classifier error breakdown,
qualitative failure-case figure, inter-annotator agreement.
