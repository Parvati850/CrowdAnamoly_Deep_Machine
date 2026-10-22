# Technical Decisions

---

## 1. Why motion features instead of raw frames?

With only 17 labelled decodable videos, feeding raw pixel data into any deep
learning model would immediately overfit. Dense optical flow (Farneback) and
frame-difference statistics extract meaningful, compact representations of
crowd motion without requiring a large dataset.

**Abnormal crowds** tend to show: higher flow magnitude, more chaotic
directions (high entropy), larger pixel changes.
**Normal crowds** show: lower, more organised flow.

---

## 2. Why Random Forest as the baseline?

- Works well with small datasets (9 training samples).
- Provides probability outputs (needed for anomaly scoring).
- `class_weight='balanced'` handles mild class imbalance.
- No GPU required.
- Easily interpretable via feature importances.
- Fast to train (< 1 second on this dataset).

Alternatives considered:
- SVM — also available via config change.
- Logistic Regression — available but underpowered for non-linear patterns.
- Deep learning (LSTM, 3D-CNN) — deferred due to tiny dataset size.

---

## 3. Why StandardScaler inside the Pipeline?

Ensures the scaler is fitted only on training data and automatically applied
to validation and test data. Prevents data leakage. Using `sklearn.Pipeline`
means the scaler is saved inside the model checkpoint — no separate scaler
file needed.

---

## 4. Video-level train/test split

Splitting at the frame level would put frames from the same video in both
train and test sets, making results look artificially good. All frames from
the same video are kept in one split.

---

## 5. Why Streamlit for the dashboard?

- Zero frontend code required.
- Python-native — consistent with the rest of the codebase.
- Suitable for academic demonstrations.
- Supports file upload, sliders, real-time placeholders.

---

## 6. Alert logic design

Three conditions prevent false-alarm flooding:
1. **Score threshold** — score must exceed a configurable value (default 0.55).
2. **Persistence** — must persist for N consecutive windows (default 3).
3. **Cooldown** — minimum gap between alerts (default 10 seconds).

This mirrors real-world alarm system design principles.

---

## 7. Python 3.14 compatibility

All code uses standard library features available in Python 3.10+.
The `X | Y` union type hint syntax is supported from Python 3.10.
All dependencies (`opencv-python 4.13`, `scikit-learn 1.8`, `streamlit 1.57`,
`torch 2.14`, `numpy 2.4`, `pandas 3.0`) import and run correctly on 3.14.

---

## 8. Non-decodable MOV files

Nine files fail with `this file requires zlib support compiled in`.
These are QuickTime files with a zlib-compressed container layer.
OpenCV's bundled FFmpeg does not include zlib for these specific containers.
**Decision:** Exclude these files from training. They represent 3 normal and
6 extra videos. The remaining 17 labelled decodable videos are sufficient for
a baseline demonstration.
