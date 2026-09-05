# Workflow diagrams (Mermaid sources — render at mermaid.live or in any
# Markdown viewer, export as PNG/SVG for the dissertation)

## D1 — Two-stage inference system (Plan A, the recommended design)

```mermaid
flowchart LR
    A[Smartphone photo\n640x640] --> B["Stage 1: Detector\nYOLO26m (11 classes)\ntest mAP@0.5 = 0.753"]
    B --> C[Detected elements\nbounding boxes]
    C --> D[Crop each box\n+10% padding]
    D --> E["Stage 2: Classifier\nSwin-Tiny (binary)\ntest accuracy = 0.948"]
    E --> F[Damaged / Undamaged\nper element]
    F --> G[Structured damage report\nper housing unit]
```

## D2 — MLOps pipeline (as implemented)

```mermaid
flowchart TB
    subgraph Data["Data layer (DVC + Google Drive)"]
        R[data/raw\n9,964 images] --> V[validate\nlabel gate]
        AN[data/annotations\n11-class labels] --> V
    end
    V --> PS["prepare_split\n70/15/15, seed 42\ndetection11 + detection22"]
    PS --> CC["crop_classification\n14,702 padded crops"]
    subgraph Train["Training (A100, 200 /30 epochs)"]
        PS --> TD["train_detection\n8 configs: n/s/m x det11/det22"]
        CC --> TC["train_classifier\ndinov2 / efficientnet / swin"]
    end
    subgraph Track["Tracking & versioning"]
        TD --> ML[(MLflow\nexperiments + registry)]
        TC --> ML
        TD --> DL[dvc.lock\nrun fingerprints]
    end
    subgraph Eval["Separate evaluation pipeline (test split, run once)"]
        TD --> ED[eval_detection\n3 finalists]
        TC --> EC[eval_classifier\nswin]
    end
    subgraph CI["CI/CD (GitHub Actions + CML)"]
        DL --> GH[push to GitHub]
        GH --> SM[smoke matrix on runner]
        SM --> CM[CML comment:\nmetrics + figures]
        GH --> RM[README auto-refresh]
    end
```

## D3 — Experiment design (what was compared)

```mermaid
flowchart TB
    Q{Research question:\nfold damage into the detector\nor use a second model?}
    Q --> PA["Plan A (two-stage)\ndet11 detector + binary classifier"]
    Q --> PB["Plan B (single-stage)\ndet22 detector: 11 classes x damaged/intact"]
    PA --> MA[6 detectors evaluated\nn/s/m scales, YOLO12 + YOLO26]
    PA --> CA[3 classifiers evaluated\nDINOv2 / EfficientNet-B0 / Swin-Tiny]
    PB --> MB[2 detectors evaluated\nYOLO12n / YOLO26n det22]
    MA --> T[Held-out test\nevaluated once]
    CA --> T
    MB --> T
    T --> W["Winner: Plan A\nYOLO26m + Swin-Tiny\neffective mAP@0.5 = 0.714 vs 0.694"]
```

## D4 — DVC dependency graph (simplified from `dvc dag`)

```mermaid
flowchart LR
    raw[data/raw.dvc] --> prep[prepare_split]
    ann[data/annotations.dvc] --> prep
    ann --> val[validate]
    prep --> t1["train_detection\n(yolo12n/26n x det11/det22)"]
    prep --> t2["train_detection_scaled\n(yolo12m/26m/12s/26s x det11)"]
    prep --> crop[crop_classification]
    crop --> t3["train_classifier\n(dinov2/efficientnet/swin)"]
    t1 --> e1[eval_detection]
    t2 --> e1
    t3 --> e2[eval_classifier]
```

The full ASCII DAG is in `dvc_dag.txt` (output of `dvc dag`).
