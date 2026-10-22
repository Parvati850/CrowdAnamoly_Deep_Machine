# User Guide

## Prerequisites

- Windows 10/11
- Python 3.11 (the project ships with a `.venv` pre-configured)
- The `.venv` folder must exist inside the project directory

---

## Using the virtual environment

All commands below use `.\.venv\Scripts\python.exe` (Python 3.11) and the
venv's own `streamlit.exe`. This avoids stale module caches that can cause
errors like `TypeError: AlertManager.__init__() got an unexpected keyword
argument 'source_name'` when the system Python loads an outdated `.pyc`.

**Activate the venv (optional but recommended):**
```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"
.\.venv\Scripts\Activate.ps1
```

After activation you can use `python` and `streamlit` directly.
Without activation, prefix every command with `.\.venv\Scripts\`.

---

## Step 1 — Install / verify dependencies

```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"
.\.venv\Scripts\pip.exe install -r requirements.txt
```

All packages are already pinned in `requirements.txt`. Expected output: all
requirements already satisfied (no downloads needed if venv is intact).

---

## Step 2 — Inspect the dataset

```powershell
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
```

Expected output:
```
Phase 1: Dataset Inspection
Inventory complete. Total=43  Normal=12  Abnormal=8  Extra=23  Decodable=34
Manifest saved to dataset/dataset_manifest.csv
DATASET_REPORT.md written.
```

Produces: `dataset/dataset_manifest.csv`, `docs/DATASET_REPORT.md`,
`outputs/reports/validation_findings.json`

---

## Step 3 — Prepare dataset (split + extract features)

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
```

Expected output:
```
train → 9 videos  (abnormal=4, normal=5)
val   → 3 videos  (abnormal=2, normal=1)
test  → 5 videos  (abnormal=2, normal=3)
Features saved: features_train.npz  shape=(9, 33)
Features saved: features_val.npz    shape=(3, 33)
Features saved: features_test.npz   shape=(5, 33)
```

Takes 2–3 minutes (dense optical flow on all videos).

---

## Step 4 — Train the model

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py
```

Expected output:
```
Training baseline on 9 samples (33 features)...
Training complete.
Model saved to models/baseline_model.pkl
Training-set accuracy: 1.000
```

Train-set accuracy = 1.0 is expected for a Random Forest on 9 samples.
This is NOT a measure of generalisation — evaluate on the held-out test set.

---

## Step 5 — Evaluate on the test set

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py
```

Expected output:
```
Balanced accuracy : 0.5833
ROC-AUC           : 0.5833
TP=1  TN=2  FP=1  FN=1
```

With only 5 test videos these metrics are indicative only. The model
performs near chance level — this is honest and documented.

---

## Step 6 — Run quick inference spot-check

```powershell
.\.venv\Scripts\python.exe scripts/quick_infer.py
```

Runs the trained model on 5 test videos and prints predicted vs true labels.

---

## Step 7 — Launch the dashboard

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Opens at: **http://localhost:8501**

### Dashboard pages
| Page | What it does |
|------|-------------|
| **Overview** | Dataset statistics, label distribution charts, system status |
| **Video Analysis** | Upload a video → see score, timeline, metadata |
| **Live Monitor** | Webcam or RTSP real-time inference |
| **Event Log** | All fired alerts with timestamps, ack/dismiss controls |
| **Model Performance** | Evaluation metrics, confusion matrix, ROC curve |
| **About** | Project description, pipeline explanation, limitations |

---

## Step 8 — Run automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: **59 passed, 0 failed**

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `TypeError: AlertManager.__init__() got an unexpected keyword argument 'source_name'` | You are using system Python, not the venv. Use `.\.venv\Scripts\streamlit.exe run app.py`. Also restart the server — it may be caching an old module. |
| `Model checkpoint not found` | Run `train_model.py` first |
| `Manifest not found` | Run `inspect_dataset.py` first |
| `split column missing` | Run `prepare_dataset.py` first |
| OpenCV codec warnings in terminal | Normal — OpenCV prints MSMF warnings to stderr; they do not affect results |
| Video upload fails to decode | File may use an unsupported codec. Try `.avi` (MJPEG) or `.mp4` (H.264) |
| Dashboard shows "no model" | Run the 4 scripts in order first |
| Webcam not found | Check webcam index in `config/config.yaml` (default: 0). Try 1 or 2. |
| `InconsistentVersionWarning` when loading model | Model was saved with a different sklearn version. Re-run `train_model.py` to create a fresh checkpoint with the current venv. |

---

## Changing settings

All settings live in `config/config.yaml`. Key ones:

| Setting | Default | Effect |
|---------|---------|--------|
| `alerts.threshold` | 0.55 | Score ≥ this triggers alert. Change in sidebar slider too. |
| `alerts.persistence` | 3 | Consecutive suspicious windows before alert fires |
| `alerts.cooldown_seconds` | 10 | Minimum seconds between alerts |
| `preprocessing.frame_sample_interval` | 10 | Sample every Nth frame |
| `model.baseline_classifier` | random_forest | Also: `svm`, `logistic` |

After changing the classifier, re-run `train_model.py` and `evaluate_model.py`.

---

## Re-running the full pipeline from scratch

```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
.\.venv\Scripts\python.exe scripts/train_model.py
.\.venv\Scripts\python.exe scripts/evaluate_model.py
.\.venv\Scripts\python.exe scripts/quick_infer.py
.\.venv\Scripts\streamlit.exe run app.py
```

---

## Verifying the Video Analysis workflow (end-to-end debug)

```powershell
.\.venv\Scripts\python.exe scripts/debug_video_analysis.py
```

This runs all 10 steps of the Video Analysis page logic against the real
dataset and model, and reports PASS/FAIL for each step without a browser.
