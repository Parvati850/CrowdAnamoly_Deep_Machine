# Real-Time AI-Based Crowd Anomaly Detection

An academic mini-project that analyses crowd video footage for unusual motion
patterns using computer vision and machine learning.

> **Important:** This is a decision-support prototype for educational purposes.
> It detects unusual motion patterns — not confirmed emergencies, panic, or injuries.
> Every alert requires human review before any action is taken.

---

## Quick Start (Windows PowerShell)

```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"

# Install dependencies
pip install -r requirements.txt

# Step 1 — Inspect dataset and build manifest
python scripts/inspect_dataset.py

# Step 2 — Split videos and extract motion features (~2 min)
python scripts/prepare_dataset.py

# Step 3 — Train baseline Random Forest
python scripts/train_model.py

# Step 4 — Evaluate on held-out test set
python scripts/evaluate_model.py

# Step 5 — Spot-check inference on test videos
python scripts/quick_infer.py

# Step 6 — Launch dashboard
python -m streamlit run app.py
```

Dashboard opens at **http://localhost:8501**

---

## Project Overview

| Item | Detail |
|------|--------|
| Language | Python 3.10+ (tested on 3.14.0) |
| ML model | StandardScaler → RandomForestClassifier (100 trees) |
| Features | Dense optical flow + frame difference statistics (33 values per video) |
| Dataset | 43 crowd videos — 17 labelled decodable used for training |
| Test balanced accuracy | **0.583** on 5-video held-out test set |
| Dashboard | Streamlit — 6 pages |

---

## Project Structure

```
CrowdAnomalyDetection/
├── app.py                          ← Streamlit dashboard (6 pages)
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   ├── config.yaml                 ← All configurable settings
│   ├── Normal Crowds/              ← Labelled normal crowd videos
│   └── Abnormal Crowds/            ← Labelled abnormal crowd videos
├── dataset/
│   ├── dataset_manifest.csv        ← Generated video inventory + splits
│   └── Crowd Dataset - extra/      ← Unlabelled extra videos (not used for training)
├── src/
│   ├── data/
│   │   ├── inventory.py            ← Video discovery, metadata, label assignment
│   │   ├── validate.py             ← Dataset quality checks
│   │   ├── extract_frames.py       ← Frame sampling utilities
│   │   └── split_dataset.py        ← Video-level train/val/test split
│   ├── features/
│   │   ├── motion_features.py      ← Optical flow + frame diff → 33-value vector
│   │   └── video_features.py       ← Feature matrix builder + cache
│   ├── models/
│   │   ├── baseline.py             ← sklearn Pipeline (scaler + RF)
│   │   ├── train.py                ← Training orchestrator
│   │   ├── evaluate.py             ← Metrics, confusion matrix, ROC
│   │   └── predict.py              ← Single-video inference
│   ├── inference/
│   │   ├── video_inference.py      ← Windowed video analysis, cap in finally
│   │   ├── realtime.py             ← Webcam/RTSP threaded inference
│   │   └── alerts.py               ← AlertManager with unique IDs + source_name
│   ├── dashboard/
│   │   ├── components.py           ← Streamlit UI components
│   │   └── visualizations.py       ← Plotly chart helpers
│   └── utils/
│       ├── paths.py                ← Path resolution, config loading
│       └── logging_config.py       ← Logging setup
├── scripts/
│   ├── inspect_dataset.py          ← Phase 1: dataset inspection
│   ├── prepare_dataset.py          ← Phase 3+4: split + features
│   ├── train_model.py              ← Phase 5: training
│   ├── evaluate_model.py           ← Phase 6: evaluation
│   ├── quick_infer.py              ← Quick spot-check inference
│   ├── test_app_pages.py           ← Dashboard page smoke tests
│   └── convert_unreadable_videos.py ← Optional: FFmpeg conversion of bad MOVs
├── tests/                          ← 51 automated tests (all passing)
│   ├── test_alerts.py              ← 14 tests incl. unique IDs, ack/dismiss
│   ├── test_video_analysis.py      ← 10 tests incl. dedup, threshold boundary
│   ├── test_evaluate.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_inventory.py
│   ├── test_model.py
│   └── test_split.py
├── models/
│   └── baseline_model.pkl          ← Trained model checkpoint
├── generated_data/
│   └── features/                   ← Cached feature vectors (.npz)
├── outputs/
│   ├── reports/                    ← evaluation_metrics.json, train_metadata.json
│   ├── plots/                      ← confusion_matrix.png, roc_curve.png
│   └── logs/                       ← app.log, alerts.csv
└── docs/
    ├── PROJECT_ASSESSMENT.md
    ├── DATASET_REPORT.md
    ├── CODEC_DIAGNOSIS.md          ← Root-cause analysis of non-decodable MOVs
    ├── IMPLEMENTATION_PLAN.md
    ├── TECHNICAL_DECISIONS.md
    ├── SYSTEM_ARCHITECTURE.md
    ├── USER_GUIDE.md
    └── VIVA_PREPARATION.md
```

---

## Pipeline (beginner-friendly)

```
VIDEO INPUT
  ↓ Upload file or open webcam
FRAME DECODING
  ↓ OpenCV reads frames; every 10th frame is kept; resized to 224×224
MOTION FEATURE EXTRACTION
  ↓ Dense optical flow (Farneback) between consecutive frames
  ↓ Measures speed, direction chaos, pixel change → 33 numbers
MODEL PREDICTION
  ↓ StandardScaler normalises the 33 values
  ↓ Random Forest outputs P(abnormal) in [0, 1]
THRESHOLD DECISION
  ↓ Score ≥ threshold (default 0.55) → suspicious
  ↓ After N consecutive suspicious windows + cooldown → ALERT
HUMAN REVIEW
  A human operator decides whether action is needed
```

---

## Dataset

| Folder | Label | Total | Decodable | Used |
|--------|-------|-------|-----------|------|
| `config/Normal Crowds/` | normal | 12 | 9 | Training |
| `config/Abnormal Crowds/` | abnormal | 8 | 8 | Training |
| `dataset/Crowd Dataset - extra/` | unknown | 23 | 17 | Not used |

**9 non-decodable MOV files** use a zlib-compressed QuickTime container not
supported by OpenCV on Windows. See `docs/CODEC_DIAGNOSIS.md` for full details.
Use `scripts/convert_unreadable_videos.py` to attempt conversion with FFmpeg.

---

## Actual Evaluation Results (held-out test set, 5 videos)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Balanced Accuracy | **0.583** | Marginally above chance (0.5) |
| ROC-AUC | **0.500** | Score ordering is essentially random |
| TP / TN / FP / FN | 1 / 2 / 1 / 1 | On only 5 test videos |

These results are honest and expected. 17 training videos is far too few
for reliable generalisation. The pipeline is correct; the dataset is tiny.

---

## Running Tests

```powershell
python -m pytest tests/ -v
# Expected: 51 passed, 0 failed
```

---

## Key Configuration (`config/config.yaml`)

| Setting | Default | Description |
|---------|---------|-------------|
| `preprocessing.frame_sample_interval` | 10 | Sample every Nth frame |
| `preprocessing.target_width/height` | 224 | Resize target |
| `alerts.threshold` | 0.55 | Score ≥ this triggers alert |
| `alerts.persistence` | 3 | Consecutive windows before alert fires |
| `alerts.cooldown_seconds` | 10 | Minimum gap between alerts |
| `model.baseline_classifier` | random_forest | Also: svm, logistic |

---

## Limitations

- Very small dataset — results are not generalisable.
- Detects unusual motion only, not panic, injury, or intent.
- Non-decodable MOV files excluded (codec limitation, not data corruption).
- Not suitable for real safety-critical deployment.
- No face recognition, biometric identification, or demographic profiling.

---

## Future Enhancements

- Larger annotated dataset (UCF-Crime, ShanghaiTech).
- Temporal deep learning (LSTM / 3D-CNN).
- Crowd density estimation and heatmap overlays.
- Multi-camera fusion and scene-context features.

---

## Changes Since Initial Build (Audit Phase)

| Area | Fix |
|------|-----|
| `alerts.py` | Added unique `event_id` per event; `source_name` constructor param; `acknowledge`/`dismiss` by ID |
| `video_inference.py` | `cap.release()` now in `try/finally`; alert call no longer passes `source=` per-call |
| `realtime.py` | Removed `source=str(source)` per-call — uses manager's `source_name` |
| `app.py` | Session-state guard prevents re-processing on reruns; temp file cleanup in all paths; video metadata shown before analysis; playback preview; `source_name` set to filename; Live Monitor camera pre-check; Event Log ack/dismiss by ID; Model Performance per-video breakdown |
| `motion_features.py` | Docstring corrected (was "34 values", actually 33) |
| Tests | 18 new tests added (51 total); covers threshold boundary, unique IDs, source_name, ack/dismiss, corrupt video, dedup |
