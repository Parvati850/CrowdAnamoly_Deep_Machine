# Implementation Plan

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Workspace & dataset inspection | ✅ Complete |
| 2 | Python environment & dependency verification | ✅ Complete |
| 3 | Dataset manifest & validation | ✅ Complete |
| 4 | Video preprocessing & feature extraction | ✅ Complete |
| 5 | Baseline model training | ✅ Complete |
| 6 | Group-aware evaluation | ✅ Complete |
| 7 | Inference on uploaded video | ✅ Complete |
| 8 | Dashboard integration | ✅ Complete |
| 9 | Webcam / RTSP live integration | ✅ Implemented (hardware-dependent) |
| 10 | Temporal deep-learning extension | Deferred (needs larger dataset) |
| 11 | Automated testing, bug fixes, documentation | ✅ Complete |
| 12 | End-to-end validation | ✅ Complete |

---

## Commands to run the full pipeline

```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"

# Phase 1 — inspect dataset, build manifest
python scripts/inspect_dataset.py

# Phase 3+4 — split videos, extract motion features
python scripts/prepare_dataset.py

# Phase 5 — train baseline Random Forest
python scripts/train_model.py

# Phase 6 — evaluate on held-out test set, save plots
python scripts/evaluate_model.py

# Phase 7 — quick inference on a few test videos
python scripts/quick_infer.py

# Phase 8 — launch the Streamlit dashboard
streamlit run app.py

# Run all automated tests
python -m pytest tests/ -v
```

---

## Feature Engineering

Each video → 33-dimensional feature vector:

- **8 base statistics** per consecutive frame pair:
  - Optical flow magnitude: mean, std, max
  - High-motion pixel fraction
  - Direction entropy (chaos measure)
  - Frame-difference: mean, std, changed-pixel fraction

- **4 aggregations** across all pairs: mean, std, max, min

- **1 count**: number of frame pairs

Total: 8 × 4 + 1 = **33 features**

---

## Model Pipeline

```
Video → Frame sampling (every 10th frame)
      → Resize to 224×224 (letterboxed)
      → Optical flow (Farneback) between consecutive sampled frames
      → Frame-difference statistics
      → Aggregate over all pairs → 33-dim vector
      → StandardScaler (fit on train only)
      → RandomForestClassifier (100 trees, class_weight='balanced')
      → P(normal), P(abnormal)
```
