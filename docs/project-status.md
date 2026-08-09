# Project status — MLOps infrastructure

**Last updated:** 2026-08-09
**Repo:** `AIabdAI/War-Damage-Assessment`, branch `master`

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
