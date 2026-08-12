# Project status — MLOps infrastructure

**Last updated:** 2026-08-12
**Repo:** `AIabdAI/War-Damage-Assessment`, branch `master`

## Update 2026-08-12 — two-approach comparison pipeline

The class scheme moved to **11 classes** (Beam removed; ids remapped, labels on
disk already converted — 9,964 pairs, 14,702 objects, gate-verified). The
pipeline was rebuilt to compare two damage-assessment approaches:

- **Approach A (end-to-end):** YOLO detects 22 classes = 11 base classes ×
  damaged/undamaged, encoded as `new_cls = cls + 11 * damage_flag`.
- **Approach B (two-stage):** YOLO detects the 11 base classes; separate
  classifiers (DINOv2 / EfficientNet / Swin — next phase) classify crops.

New stages (all params.yaml-driven, one shared deterministic 70/15/15 split):
`prepare_split` (detection11 + detection22 datasets + split manifest/reports)
→ `crop_classification` (14,702 padded crops into damaged/undamaged buckets)
→ `train_detection` matrix (yolo12n/yolo26n × 11/22; `--smoke` for CI).
Canonical label parsing lives in `scripts/label_common.py` (shared by both
data stages so objects stay aligned; contract locked by tests).
MLflow experiment `war-damage-detection` tags each run approach=A|B, logs
`classes_map.json` so every run is self-describing, and registers weights as
`{model}-det{variant}`. CML posts a 4-run comparison table + figures on PRs
and auto-refreshes the README "Latest Results" section on pushes to master.
Full (non-smoke) training is reserved for the A100 server: `dvc repro`.

## What is set up and verified working

| Component | State |
|---|---|
| **DVC pipeline** (`dvc.yaml`) | 3 stages: validate → prepare → train. Runs green end-to-end; first `dvc.lock` committed (this is the baseline `dvc metrics diff` compares against). |
| **Data versioning** | `data/raw.dvc` (8,861 images, 848 MB) + `data/annotations.dvc` (9,106 files) on the gdrive remote `storage`. All cache pushed. |
| **Params** | `params.yaml` is the single source of truth (12 classes, `train_smoke`: 10 epochs, yolo12n + yolo26n). Both `prepare_smoke_dataset.py` and `train_smoke.py` read it. |
| **MLflow** | `mlflow.db` (sqlite, local, gitignored). Experiment `war-damage-smoke`: 2 registered baselines + smoke runs logged automatically by every `dvc repro`. |
| **CI: tests** | `.github/workflows/tests.yml` — pytest on every push/PR. 6 label-contract tests (`tests/test_labels.py` + root `conftest.py`). |
| **CI: data guard** | `.github/workflows/data-guard.yml` — blocks any real data file under `data/` in git (only `.dvc` pointers allowed). |
| **CI: CML** | `.github/workflows/cml.yml` — on every PR: `dvc pull` (service account) → `dvc repro` → posts PR comment with metrics diff vs master + training curves, confusion matrices, GT-vs-prediction images. |
| **Secrets** | `GDRIVE_SA_JSON` set on GitHub (service account `dvc-ci@war-damage-dvc.iam.gserviceaccount.com`, Viewer on the Drive folder). Key verified against the remote before upload. |
| **Docker** | `Dockerfile` (dev, py3.12-slim) + `Dockerfile.train` (CUDA 12.4) + `docker-compose.yml` (`dev` and GPU `train` services). |

## Current smoke metrics (300-image subset, 10 epochs — plumbing check, not model quality)

| model | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| yolo12n | 0.256 | 0.148 | 0.661 | 0.219 |
| yolo26n | 0.155 | 0.092 | 0.656 | 0.088 |

Registered baselines (earlier 10-epoch run, different data state): yolo12n 0.601 / yolo26n 0.624 mAP50.
The gap is expected — the smoke subset caps at 300 images (`prepare_smoke_dataset.py --max-images`).

## Data fixes applied 2026-08-09

- 23 label files (`tile_*`, `bricks_*`) had boxes with w/h slightly > 1.0 (tiling
  float artifacts, max 1.037). Clamped to image bounds by
  `scripts/clamp_label_boxes.py` (kept in git as documentation).
- Validation gate (`verify_labels.py`) now skips hidden tool-state files
  (`.annotator_progress.txt`).
- Data pointers grew: raw 6,440 → 8,861 files; annotations 6,782 → 9,106
  (data that was on disk but never committed to the pointers). Pushed to Drive.

## Known notes / next steps

- **Environment rule:** always use `.venv` (Python 3.12, DVC 3.67). System DVC 2.x
  cannot read this repo, and `dvc repro` uses whatever `python` is on PATH.
- CML has not produced its first real PR comment yet — open any PR to exercise it.
- The service account is **Viewer** (pull-only). If CI ever needs to `dvc push`,
  move the folder to a Shared Drive or bump the role to Editor.
- Full training (all clean pairs, more epochs, A100) is the next phase — raise
  `--max-images 0` in prepare + a proper training config.
- 831 label files carry box-extent warnings (box edge slightly outside the image);
  informational only, the cropper clamps them.
