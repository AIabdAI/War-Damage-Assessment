# MLOps guide — how to run and use the tools

**Rule zero: always work inside the venv.** System DVC 2.x cannot read this repo,
and DVC runs pipeline stages with whatever `python` is on PATH.

```powershell
.\.venv\Scripts\activate
```

## The core loop — DVC pipeline

```powershell
dvc repro          # reruns ONLY the stages whose inputs changed
```

- Change training settings in `params.yaml` (e.g. `epochs: 20`) → only the train
  stage reruns.
- Change labels in `data/annotations` → validate, prepare and train all rerun.
- Nothing changed → nothing runs. The pipeline is the single way to produce results.

After a successful run, compare and publish:

```powershell
dvc metrics show               # current metrics (reports/metrics.json)
dvc metrics diff               # vs the last committed dvc.lock
git add dvc.lock reports/metrics.json params.yaml
git commit -m "exp: 20 epochs"
git push                       # git carries pointers + lock
dvc push                       # Drive carries the actual data/weights
```

`git push` and `dvc push` go together — one without the other leaves the repo
and the remote out of sync.

## MLflow — comparing experiments

Every `dvc repro` logs one run per model into the `war-damage-smoke` experiment.

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 → select runs → **Compare** for metrics tables,
per-epoch curves, and parameters side by side (baselines included).

## CML — automated PR reports

Runs on GitHub, not locally. Flow:

```powershell
git checkout -b exp/my-change     # e.g. bump epochs in params.yaml
git commit -am "exp: ..."
git push -u origin exp/my-change
# open the PR on GitHub
```

The `cml-report` workflow then: pulls data with the service account →
`dvc repro` → posts a PR comment containing the metrics diff vs master,
training curves, normalized confusion matrix, and ground-truth vs prediction
images for each model.

Requirements (already configured): `GDRIVE_SA_JSON` secret on the repo;
the service account has Viewer access to the Drive data folder.

## Tests

```powershell
pytest tests/          # label-contract tests; CI runs them on every push
```

## Docker

```powershell
docker compose run dev     # clean Python 3.12 environment
docker compose run train   # CUDA image, runs `dvc repro train` (for GPU boxes / A100)
```

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

## Pulling everything on a fresh machine

```powershell
git clone https://github.com/AIabdAI/War-Damage-Assessment.git
cd War-Damage-Assessment
python -m venv .venv; .\.venv\Scripts\activate
pip install -r requirements-train.txt
dvc pull            # needs Drive access (OAuth prompt or service account)
dvc repro
```

## See also

- [project-status.md](project-status.md) — what is set up, current metrics, next steps
- [dvc-vs-mlflow.md](dvc-vs-mlflow.md) — why the project uses both trackers
