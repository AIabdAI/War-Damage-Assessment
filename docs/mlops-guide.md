# MLOps guide — managing the project, training, and publishing results

**Rule zero: always work inside the venv.** System DVC 2.x cannot read this repo,
and DVC runs pipeline stages with whatever `python` is on PATH.

```powershell
.\.venv\Scripts\activate        # Windows PC
# source .venv/bin/activate     # Linux server
```

## 1. Managing the project — the daily loop

Everything flows through three files:

| file | role |
|---|---|
| `params.yaml` | ALL knobs: classes, split ratios/seed, crop padding, epochs, models |
| `dvc.yaml` | the pipeline: validate → prepare_split → crop_classification → train matrix |
| `dvc.lock` | what was actually run (data hashes + params + outputs) — committed to git |

```powershell
dvc status                      # what changed / what would rerun
dvc repro                       # rerun ONLY the affected stages
```

- Change a parameter → `dvc repro` reruns only affected stages.
- Change labels (annotation tool) → `dvc commit data/annotations.dvc` then
  `dvc repro` — split, both detection variants, and crops all regenerate.
- Compare experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
  → http://127.0.0.1:5000, experiment `war-damage-detection`
  (runs tagged approach=A|B, variant=11|22, model).

Pipeline stages (all params.yaml-driven, one shared deterministic 70/15/15 split):

- `prepare_split` → `data/processed/detection11` (Approach B) +
  `detection22` (Approach A, `new_cls = cls + 11 * damage_flag`) + reports
- `crop_classification` → `data/processed/classification/{split}/{damaged,undamaged}`
- `train_detection@{model}-{variant}` → 4 configs: {yolo12n, yolo26n} × {11, 22}

## 2. Training on the PC — smoke / sanity only

The laptop (4 GB GPU, ~16 GB RAM) verifies; it does not produce:

```powershell
# plumbing check, ~2 min: 1 epoch on 5% of data.
# Metrics WILL be ~0 - that is expected and fine.
python scripts/train_detection.py --model yolo12n --variant 22 --smoke

# "does it learn?" sanity run, ~30-40 min (verified: mAP50 ~0.48 by epoch 4)
python scripts/train_detection.py --model yolo12n --variant 22 --epochs 5 --batch 8
```

PC rules: use `--batch 8` (batch 16 on the full dataset exhausted RAM),
close heavy apps, and do NOT run the full train stages via `dvc repro`
here — 4 configs × 100 epochs belongs on the server.

## 3. Training on the server (A100) — the real runs

```bash
git clone https://github.com/AIabdAI/War-Damage-Assessment.git && cd War-Damage-Assessment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt

# Drive remote auth via the service account key (same one CI uses):
dvc remote modify --local storage gdrive_use_service_account true
dvc remote modify --local storage gdrive_service_account_json_file_path /path/to/war-damage-dvc-*.json

dvc pull            # data + labels
dvc repro           # full pipeline: gate -> split -> crops -> all 4 trainings (100 epochs)
```

`dvc repro` fills the four `train_detection` entries in `dvc.lock` — the
auditable record of exactly what data/params produced which weights.

Notes:
- The service account is **Viewer** on the Drive folder. For the server to
  `dvc push` weights back, bump it to **Editor** (or push with your own OAuth).
- Docker alternative: `docker compose run train` (CUDA image, runs `dvc repro train`).

## 4. Publishing results to GitHub

Results travel by two roads — **always do both**:

```bash
dvc push                                       # bytes: weights + datasets -> Google Drive
git add dvc.lock reports/ params.yaml
git commit -m "train: full 100-epoch runs on A100"
git push origin master
```

- `git push` carries the *record* (`dvc.lock`, `reports/metrics_*.json`,
  figures) and triggers CI: a push to master auto-refreshes the
  **"## Latest Results"** section of README.md (`[skip ci]`-guarded commit).
- `dvc push` carries the *bytes*: any machine can then `dvc pull` the weights.
- For reviewed changes use a branch + PR: the CML workflow smoke-trains all
  4 configs and posts a comparison table + curves/confusion-matrix/prediction
  images as a PR comment. (Smoke numbers in that table are ~0 by design —
  their only signal is "the pipeline runs".)

**MLflow is per-machine**: `mlflow.db` is local, gitignored SQLite. Server runs
land in the server's db — copy `mlflow.db` back to the PC after a big run
(single file), or set up a shared tracking server later.

## When you add or fix data

```powershell
dvc status                          # shows data/annotations changed
dvc commit data/annotations.dvc     # update the pointer to match disk
git add data/annotations.dvc
git commit -m "data: ..."
git push
dvc push
```

The `data-guard` CI blocks any real data file committed to git — only `.dvc`
pointers belong there.

## Tests

```powershell
pytest tests/     # label-contract + canonical-parse tests; CI runs them on every push
```

## See also

- [project-status.md](project-status.md) — what is set up, current state, next steps
- [dvc-vs-mlflow.md](dvc-vs-mlflow.md) — why the project uses both trackers
