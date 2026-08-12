# Server runbook — full training on the A100

Every step from a bare server to published results, including the Google
Drive credential setup. Companion to [mlops-guide.md](mlops-guide.md).

## 0. Prerequisites (once)

- NVIDIA driver working: `nvidia-smi` shows the A100
- Python >= 3.10, git, internet access

## 1. Clone + environment

```bash
git clone https://github.com/AIabdAI/War-Damage-Assessment.git
cd War-Damage-Assessment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# must print: True NVIDIA A100...
```

## 2. Google Drive key (does NOT come with the clone)

The DVC remote (`storage`, folder id `1Ts4w602gk_PLeDyVPC-eqEIZq0iAFBLa`) is in
the committed `.dvc/config`, but the credential config lives in the gitignored
`.dvc/config.local` — set it up once per machine.

From the PC, copy the service-account key over:

```powershell
scp C:\Users\abd50\war-damage-dvc-28baf1d71169.json user@SERVER:~/gdrive-sa.json
```

On the server, inside the repo:

```bash
dvc remote modify --local storage gdrive_use_service_account true
dvc remote modify --local storage gdrive_service_account_json_file_path ~/gdrive-sa.json
```

Rules:
- Keep the key OUTSIDE the repo folder (`~/gdrive-sa.json`), never commit it.
- `--local` is mandatory — it keeps the setting out of the committed config.
- **Before training**: in Drive, share the data folder with
  `dvc-ci@war-damage-dvc.iam.gserviceaccount.com` as **Editor** (it is Viewer
  by default, which allows `dvc pull` but not the `dvc push` of results).
- Never use the interactive OAuth flow on a headless server (it wants a browser).

## 3. Pull the data

```bash
dvc pull        # ~1 GB: 9,964 image/label pairs; no prompt if step 2 is correct
```

## 4. Run the full pipeline

```bash
tmux new -s train      # survive SSH disconnects; detach Ctrl+B D, reattach: tmux attach -t train
dvc repro
```

Order: validate → prepare_split → crop_classification →
4 detection trainings (yolo12n/yolo26n × 11/22, 100 epochs) →
3 classifiers (dinov2/efficientnet/swin, 30 epochs).

Optional A100 tuning in `params.yaml` before starting (defaults are
laptop-sized): `train_detection.batch: 64`, `train_classifier.batch: 128`.
Commit the change together with the results.

## 5. Check results

```bash
dvc metrics show
cat reports/metrics_*.json
```

MLflow runs land in the SERVER's local `mlflow.db`
(experiments `war-damage-detection`, `war-damage-classification`).

## 6. Publish — both roads, always

```bash
dvc push                          # weights + datasets -> Drive (needs Editor)
git add dvc.lock reports/ params.yaml
git commit -m "train: full runs on A100 (4 detection + 3 classifiers)"
git push origin master
```

The push triggers CI; the README "Latest Results" section refreshes itself
via a `[skip ci]` bot commit.

## 7. Back on the PC

```bash
git pull --rebase origin master   # dvc.lock, metrics, README bot commit
dvc pull                          # trained weights, if wanted locally
# optional - server MLflow runs into the local UI (OVERWRITES local mlflow.db,
# back it up first):
scp user@SERVER:~/War-Damage-Assessment/mlflow.db .
```
