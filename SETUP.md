# Download and run this project on any device

Four levels, from "just look at the results" to "reproduce the full training".
Pick the one you need — each builds on the previous.

| level | what you get | needs | time |
|---|---|---|---|
| [0](#level-0--read-the-results) | reports, metrics, figures | git only | 2 min |
| [1](#level-1--run-the-models-on-your-own-photos) | run the models on your photos | Python + models | 15 min |
| [2](#level-2--serve-the-models-to-the-mobile-app) | model-distribution API | + FastAPI | 5 min |
| [3](#level-3--reproduce-the-pipeline) | full training / evaluation | + dataset + GPU | hours |

---

## Level 0 — read the results

```bash
git clone https://github.com/AIabdAI/War-Damage-Assessment.git
cd War-Damage-Assessment
```

Everything below is plain text/images in the repo — no setup at all:

| file | contents |
|---|---|
| `README.md` | auto-generated results tables (validation + held-out test) |
| `reports/comparison_report.md` | full model comparison, final verdict |
| `reports/compression_report.md` | mobile compression results |
| `docs/mlops-guide.md` | how the pipeline works day to day |
| `All Information/` | dissertation materials: tables, figures, diagrams |

---

## Level 1 — run the models on your own photos

### 1.1 Install Python 3.10+ and create an environment

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Windows (Git Bash)
python -m venv .venv && source .venv/Scripts/activate
```

### 1.2 Install dependencies

```bash
pip install -r requirements-train.txt -r requirements-mobile.txt
```

CPU-only machine? That is fine for inference. Only training needs a GPU.

### 1.3 Get the models

The trained weights are stored in DVC (Google Drive), not in git.

First install DVC 3 (2.x cannot read this repo's pointer files):

```bash
pip install "dvc[gdrive]>=3.0"
```

**Authentication** — the remote is the Drive folder
`1Ts4w602gk_PLeDyVPC-eqEIZq0iAFBLa`. Either:

*A. Service-account key* (what the CI and training servers use — ask the
project owner for `gdrive-sa.json`):

```bash
dvc remote modify --local storage gdrive_use_service_account true
dvc remote modify --local storage gdrive_service_account_json_file_path /path/to/gdrive-sa.json
```

*B. Personal Google login* — run `dvc pull` with no extra config and sign in
through the browser once; the token is cached. Your account needs access to
the folder.

A service account can only **download**; uploading requires a personal
account (Google blocks service-account writes to personal Drives).

**Download only what you need:**

| you want | command | size |
|---|---|---|
| mobile models (enough for inference) | `dvc pull models_mobile` | 482 MB |
| best detector, PyTorch weights | `dvc pull runs_detection/yolo26m_det11` | 93 MB |
| best classifier, PyTorch weights | `dvc pull runs_classification/swin` | 210 MB |
| original images + labels | `dvc pull data/raw.dvc data/annotations.dvc` | 1.0 GB |
| training-ready detection dataset | `dvc pull data/processed/detection11` | 1.0 GB |
| damage-classification crops | `dvc pull data/processed/classification` | 649 MB |
| MLflow experiment databases | `dvc pull mlflow_pod1.db mlflow_pod2.db` | 9 MB |
| everything | `dvc pull` | ~5 GB |

Verify: `dvc status -c` should print
`Cache and remote 'storage' are in sync.`

**No credentials at all?** Ask the owner for the `models_mobile/` folder
directly, or point your client at a machine running the API (Level 2).

### 1.4 Run the two-stage assessment on an image

Save this as `predict.py` in the repository root:

```python
"""Detect building elements, then classify each as damaged/undamaged."""
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = ["Brick_Wall", "Column", "Staircase", "Floor_Tiles", "Sink",
           "Wall_Cabinet", "Window", "Door", "Air_Conditioner",
           "Light_Fixture", "Toilet"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

det = ort.InferenceSession("models_mobile/detector/yolo26m_det11/int8_dynamic.onnx",
                           providers=["CPUExecutionProvider"])
cls = ort.InferenceSession("models_mobile/classifier/swin/int8_static.onnx",
                           providers=["CPUExecutionProvider"])

img = Image.open(sys.argv[1]).convert("RGB")
W, H = img.size

# --- stage 1: detection (input 640x640, RGB, /255) ---
x = np.asarray(img.resize((640, 640)), np.float32).transpose(2, 0, 1)[None] / 255.0
out = det.run(None, {det.get_inputs()[0].name: x})[0][0]   # [x1,y1,x2,y2,conf,cls]

for x1, y1, x2, y2, conf, cid in out[out[:, 4] > 0.25]:
    # boxes are in 640x640 space -> scale back to the original image
    box = (x1 / 640 * W, y1 / 640 * H, x2 / 640 * W, y2 / 640 * H)
    pad_w, pad_h = (box[2] - box[0]) * 0.1, (box[3] - box[1]) * 0.1
    crop = img.crop((max(0, box[0] - pad_w), max(0, box[1] - pad_h),
                     min(W, box[2] + pad_w), min(H, box[3] + pad_h)))

    # --- stage 2: damage classification (224x224, ImageNet normalisation) ---
    w, h = crop.size
    s = 256 / min(w, h)
    crop = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = crop.size
    left, top = (w - 224) // 2, (h - 224) // 2
    arr = np.asarray(crop.crop((left, top, left + 224, top + 224)), np.float32) / 255.0
    arr = ((arr - MEAN) / STD).transpose(2, 0, 1)[None]
    logits = cls.run(None, {cls.get_inputs()[0].name: arr})[0][0]

    state = "DAMAGED" if logits.argmax() == 0 else "intact"
    print(f"{CLASSES[int(cid)]:16s} conf={conf:.2f}  ->  {state}")
```

```bash
python predict.py path/to/your/photo.jpg
```

Real output from this script on test images (verified 2026-09-05, running
the compressed INT8 models on CPU):

```
$ python predict.py Crack-01.jpg
Brick_Wall       conf=0.98  ->  DAMAGED

$ python predict.py particle-board-panel.jpg
Wall_Cabinet     conf=0.91  ->  intact

$ python predict.py room_photo.jpg
Air_Conditioner  conf=0.74  ->  intact
```

An image with nothing recognisable above the 0.25 confidence threshold simply
prints nothing.

The models are the ones measured in `reports/compression_report.md`:
detection mAP@0.5 0.710, damage classification 94.5 % accuracy, ~49 MB total.

---

## Level 2 — serve the models to the mobile app

```bash
python serve/build_registry.py                       # scans models_mobile/
uvicorn serve.model_api:app --host 0.0.0.0 --port 8000
```

Check it:

```bash
curl localhost:8000/health
curl localhost:8000/api/v1/bundle/latest
```

Full API contract, Flutter integration and VPS deployment (systemd + nginx +
TLS): **`serve/README.md`**.

---

## Level 3 — reproduce the pipeline

### 3.1 Extra requirements

- The dataset (~1 GB) — needs Drive credentials: `dvc pull`
- A GPU for training (the pipeline refuses full training on CPU by design);
  evaluation and compression run fine on CPU.

### 3.2 The three pipelines

```bash
# training: validate -> split -> crops -> 8 detectors + 3 classifiers
dvc repro

# held-out test evaluation (deliberately separate, run on purpose only)
cd evaluation && dvc repro --single-item && cd ..

# mobile compression + registry
cd compression && dvc repro --single-item && cd ..
```

Individual pieces:

```bash
python scripts/train_detection.py  --model yolo26m --variant 11 --smoke
python scripts/train_classifier.py --model swin --smoke
python scripts/evaluate_detection.py --run yolo26m_det11 --split test
python scripts/compress_models.py --all
python scripts/evaluate_compressed.py --all --splits val,test
```

`--smoke` runs 1 epoch on 5 % of the data — use it to verify the plumbing
before committing hours of GPU time.

### 3.3 Experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://127.0.0.1:5000
```

Experiments: `war-damage-detection`, `war-damage-classification`,
`war-damage-test-eval`, `war-damage-compression`.

### 3.4 Docker (alternative to a local environment)

```bash
docker compose run dev      # CPU environment
docker compose run train    # CUDA image, runs `dvc repro train`
```

---

## Troubleshooting — problems this project actually hit

| symptom | cause and fix |
|---|---|
| `dvc pull` fails, `.dvc` files unreadable | DVC 2.x cannot read DVC-3 files. `pip install "dvc[gdrive]>=3.0"`, and always use the venv's `dvc`. |
| `module 'lib' has no attribute 'GEN_EMAIL'` or `OpenSSL.crypto has no attribute 'sign'` | pyOpenSSL/cryptography mismatch. `pip install "pyopenssl==24.2.1" "cryptography==43.0.3"` (already pinned in `requirements-train.txt`). |
| `Service Accounts do not have storage quota` on `dvc push` | Google policy: service accounts cannot *upload* to a personal Drive. They can pull. Push from a machine using personal OAuth. |
| `SSL: UNEXPECTED_EOF` during pull/push | Drive throttling. Retry — DVC resumes; use `-j 4` to reduce parallelism. |
| Training silently runs on CPU and takes forever | torch built for a different CUDA than the driver. Check `python -c "import torch; print(torch.cuda.is_available())"`; install the matching build (e.g. `--index-url https://download.pytorch.org/whl/cu128`). The scripts now refuse full CPU training. |
| `Dataset images not found /datasets/C:/Users/...` | a `data.yaml` built on another machine. Already handled: the scripts rewrite the path at runtime. |
| `no data transfer registered for copying tensors` | ultralytics feeding CUDA tensors to a CPU ONNX session — pass `device="cpu"`. |
| Long job dies when the terminal closes | use `tmux` (`tmux new -s train`, detach `Ctrl+B D`) or `nohup ... &`. |

## Repository map

```
scripts/            data preparation, training, evaluation, compression
  prepare_split.py  crop_classification.py  label_common.py
  train_detection.py  train_classifier.py
  evaluate_detection.py  evaluate_classifier.py
  compress_models.py  evaluate_compressed.py  benchmark_latency.py
dvc.yaml            training pipeline          evaluation/dvc.yaml   test-set evaluation
compression/        compression pipeline       serve/                model API + deployment docs
params.yaml         single source of truth for every hyperparameter
reports/            metrics, figures, comparison + compression reports
docs/               MLOps guide, server runbook, DVC-vs-MLflow rationale
All Information/    dissertation materials
```
