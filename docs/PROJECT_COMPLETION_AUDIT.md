# PROJECT COMPLETION AUDIT
**Project:** Real-Time AI-Based Crowd Anomaly and Panic-Like Behaviour Detection  
**Audit date:** 2026-10-15  
**Auditor:** Automated senior-engineer audit — evidence-based, read-only  
**Project path:** `D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection`

---

## A. Executive Summary

The project is **substantially complete and functional** for its stated scope as
an academic prototype. The full pipeline — dataset inspection, feature extraction,
model training, evaluation, single-video inference, and the Streamlit dashboard —
has been verified to execute correctly without errors on Python 3.11 using the
project's virtual environment.

Key verified facts:
- 59 automated tests pass, 0 fail.
- The trained Random Forest model loads, predicts, and produces real scores.
- Video inference on a real dataset video (`861-13_70.mov`) completes correctly.
- Evaluation metrics match the saved predictions (confusion matrix verified).
- No hardcoded predictions, fake scores, or placeholder functions were found.
- No data leakage between training and test splits was detected.
- All 6 dashboard pages render without exceptions.

Honest limitations:
- The test set contains only 5 videos — metrics are statistically unreliable.
- Balanced accuracy = 0.583 (near chance). The model is a valid prototype baseline, not a production detector.
- 9 of 43 videos cannot be decoded (codec limitation, documented).
- Webcam live inference is implemented but cannot be hardware-verified in this audit.
- The original dataset folder `D:\7th sem\DL\Mini_Project\Datasets` does not exist — videos are located inside the project directory.

---

## B. Overall Status

**PARTIALLY COMPLETE — Functional baseline, with documented limitations**

The classification is not "incomplete" because the pipeline works end-to-end.
It is not "complete and verified" only because:
1. The dataset is extremely small (5 test videos) making reliable generalisation impossible.
2. Hardware-dependent features (webcam) could not be verified.
3. The original dataset path (`Datasets/`) is absent — videos live inside `config/` and `dataset/` within the project.

---

## C. Project Structure Audit

| Component | File(s) | Status |
|-----------|---------|--------|
| Entry point | `app.py` | ✅ COMPLETE AND VERIFIED |
| Configuration | `config/config.yaml` | ✅ COMPLETE AND VERIFIED |
| Dataset inspection | `scripts/inspect_dataset.py` | ✅ COMPLETE AND VERIFIED |
| Dataset manifest | `dataset/dataset_manifest.csv` | ✅ COMPLETE AND VERIFIED |
| Dataset validation | `src/data/validate.py` | ✅ COMPLETE AND VERIFIED |
| Video inventory | `src/data/inventory.py` | ✅ COMPLETE AND VERIFIED |
| Frame extraction | `src/data/extract_frames.py` | ✅ COMPLETE AND VERIFIED |
| Dataset splitting | `src/data/split_dataset.py` | ✅ COMPLETE AND VERIFIED |
| Feature extraction | `src/features/motion_features.py` | ✅ COMPLETE AND VERIFIED |
| Feature matrix builder | `src/features/video_features.py` | ✅ COMPLETE AND VERIFIED |
| Baseline model | `src/models/baseline.py` | ✅ COMPLETE AND VERIFIED |
| Training pipeline | `src/models/train.py`, `scripts/train_model.py` | ✅ COMPLETE AND VERIFIED |
| Evaluation | `src/models/evaluate.py`, `scripts/evaluate_model.py` | ✅ COMPLETE AND VERIFIED |
| Model serialization | `src/models/baseline.py` (save/load) | ✅ COMPLETE AND VERIFIED |
| Single-video inference | `src/models/predict.py` | ✅ COMPLETE AND VERIFIED |
| Windowed video inference | `src/inference/video_inference.py` | ✅ COMPLETE AND VERIFIED |
| Alert system | `src/inference/alerts.py` | ✅ COMPLETE AND VERIFIED |
| Real-time inference | `src/inference/realtime.py` | ✅ IMPLEMENTED — hardware not verified |
| Dashboard components | `src/dashboard/components.py` | ✅ COMPLETE AND VERIFIED |
| Dashboard charts | `src/dashboard/visualizations.py` | ✅ COMPLETE AND VERIFIED |
| Streamlit dashboard | `app.py` (6 pages) | ✅ COMPLETE AND VERIFIED |
| Dependency spec | `requirements.txt` (pinned) | ✅ COMPLETE |
| Test suite | `tests/` (59 tests) | ✅ COMPLETE AND VERIFIED |
| README | `README.md` | ✅ COMPLETE |
| Docs | `docs/` (9 files) | ✅ COMPLETE |
| Viva prep | `docs/VIVA_PREPARATION.md` | ✅ COMPLETE |

---

## D. Dataset Audit

### Actual video counts (verified by directory scan)

| Folder | Location in project | Videos | Label |
|--------|--------------------|----|-------|
| Normal Crowds | `config/Normal Crowds/` | 12 | normal |
| Abnormal Crowds | `config/Abnormal Crowds/` | 8 | abnormal |
| Crowd Dataset - extra | `dataset/Crowd Dataset - extra/` | 23 | unknown |
| **Total** | | **43** | |

**Note:** The path `D:\7th sem\DL\Mini_Project\Datasets` does not exist. The
videos were placed inside the project directory. The config correctly points to
their actual locations.

### Decodability

| Category | Total | Decodable | Non-decodable |
|----------|-------|-----------|---------------|
| Normal | 12 | 9 | 3 |
| Abnormal | 8 | 8 | 0 |
| Extra (unknown) | 23 | 17 | 6 |
| **Total** | **43** | **34** | **9** |

**Non-decodable files** (9): `81872953_l.mov`, `sfw20110024_l.mov`, `shibuya1.mov` (Normal); `1379-7_l.mov`, `1379-8_l.mov`, `81712424_l.mov`, `82409835_l.mov`, `82409905_l.mov`, `965-83_l.mov` (Extra).

**Root cause:** zlib-compressed QuickTime container headers. OpenCV's bundled FFmpeg on Windows does not include zlib for this container type. This is a known platform limitation, not data corruption. Documented in `docs/CODEC_DIAGNOSIS.md`.

**Impact on training:** 3 normal labelled videos excluded. 17 labelled decodable videos used for training (9 normal, 8 abnormal).

**Extra videos:** 23 videos in `Crowd Dataset - extra/` have no verified labels and are correctly excluded from all training and evaluation. Status: CORRECT.

---

## E. Environment and Dependency Audit

| Item | Verified version | Status |
|------|-----------------|--------|
| Python | 3.11.9 (venv) | ✅ Compatible |
| opencv-python | 5.0.0.93 | ✅ Compatible |
| numpy | 2.4.6 | ✅ Compatible |
| pandas | 3.0.6 | ✅ Compatible |
| scikit-learn | 1.9.1 | ✅ Compatible (CalibratedClassifierCV used correctly) |
| streamlit | 1.65.0 | ✅ Compatible |
| torch | 2.14.1+cpu | ✅ Present (CPU only, not actively used) |
| plotly | 7.1.0 | ✅ Compatible |
| PyYAML | 6.0.3 | ✅ Compatible |
| pytest | 9.1.1 | ✅ Compatible |

**Virtual environment:** `.venv/` present inside the project directory with Python 3.11.9.

**System Python:** 3.14.0 also present. If you run `streamlit run app.py` without activating the venv or using `.\.venv\Scripts\streamlit.exe`, you may encounter stale `.pyc` cache errors. Always use:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

**Requirements pinned:** Yes. `requirements.txt` lists exact versions matching the venv.

---

## F. Machine-Learning Pipeline Audit

### Feature extraction
- **Algorithm:** Dense optical flow (Farneback) + frame-difference statistics
- **Implementation:** `src/features/motion_features.py`
- **Feature vector:** 33 values (8 base stats × 4 aggregations + 1 count)
- **Verified:** Feature names count = 33, matches cached `.npz` shape (9,33) for train
- **No hardcoded outputs:** Confirmed by code scan and live execution
- **Preprocessing consistency:** Same `frame_sample_interval=10`, `target_width=224`, `target_height=224` used in both training and inference

### Model
- **Type:** `sklearn.pipeline.Pipeline` — `StandardScaler` → `RandomForestClassifier`
- **Scaler fitted on train only:** Yes (inside Pipeline — leakage-safe)
- **Checkpoint:** `models/baseline_model.pkl` — exists, loads, has `predict_proba`
- **Classifier type:** `random_forest` (100 trees, balanced class weights)
- **Training samples:** 9 videos (5 normal, 4 abnormal)

### Pipeline execution verified (real run):
```
scripts/debug_video_analysis.py — ALL 10 STEPS PASSED
  Video: 861-13_70.mov (720x576, 25fps, 503 frames)
  Feature vector: 33 values, no NaN/Inf
  Model: Pipeline loaded correctly
  Prediction: label=abnormal, score=0.660
  Threshold check: 0.660 >= 0.55 → flagged=True
  No duplicate alerts, temp file cleaned up
```

### No fabricated outputs: CONFIRMED
- Code scan found zero instances of `random.`, hardcoded scores, or `return 0.5`-style placeholders in any prediction path.

---

## G. Data Leakage and Evaluation Audit

| Check | Result |
|-------|--------|
| Split at video level (not frame level) | ✅ Confirmed — `split_dataset.py` splits by video |
| Videos appearing in multiple splits | ✅ 0 leakage videos detected |
| Scaler fitted on train only | ✅ Inside sklearn Pipeline |
| Test set used for threshold tuning | ✅ Not detected |
| Confusion matrix matches saved predictions | ✅ VERIFIED manually (TP=1, TN=2, FP=1, FN=1) |

### Evaluation results (5-video test set)

| Metric | Value |
|--------|-------|
| Balanced accuracy | 0.5833 |
| ROC-AUC | 0.5833 |
| TP / TN / FP / FN | 1 / 2 / 1 / 1 |

**Honest assessment:** With 5 test videos, these metrics have extremely high variance. The model performs marginally above chance. This is **expected and honest** for a 17-sample dataset. It does not indicate a bug — it indicates the dataset is too small for reliable generalisation. This limitation is documented throughout the project.

**Training accuracy = 1.0:** A Random Forest with 100 trees memorises 9 training samples perfectly. This is expected behaviour and is NOT evidence of generalisation.

---

## H. Video Inference Audit

**Tested video:** `861-13_70.mov` (from `dataset/Crowd Dataset - extra/` — decodable, 720×576, 25fps, 503 frames)

| Step | Result |
|------|--------|
| File opens | ✅ PASS |
| Metadata extracted | ✅ fps=25.0, width=720, height=576, frame_count=503 |
| Frames decoded | ✅ PASS |
| Preprocessing matches training | ✅ Same interval/dimensions |
| Feature vector computed | ✅ 33 values, no NaN |
| Model produces prediction | ✅ label=abnormal, score=0.660 |
| Score ≥ threshold | ✅ 0.660 ≥ 0.55 = True |
| Video resource released | ✅ cap.release() in try/finally |
| Temp file cleaned up | ✅ PASS |
| No duplicate alert IDs | ✅ PASS |

**Script:** `scripts/debug_video_analysis.py` — all 10 steps PASSED.

**Also verified via `scripts/quick_infer.py`:**
```
2017-420_l.mov   abnormal → abnormal  0.610  CORRECT
3452204_031_c.mov abnormal → normal   0.280  WRONG
637-147_l.mov    normal   → abnormal  0.610  WRONG
9019-13_l.mov    normal   → normal    0.160  CORRECT
3687-18_70.mov   normal   → normal    0.400  CORRECT
Correct: 3/5
```

These are real model predictions, not fabricated.

---

## I. Real-Time Processing Audit

| Check | Status |
|-------|--------|
| `RealtimeAnalyser` class implemented | ✅ `src/inference/realtime.py` |
| `VideoStream` threaded capture | ✅ Implemented with bounded queue |
| Start / stop controls | ✅ `start()` / `stop()` methods |
| Frame processing and inference | ✅ Same feature pipeline as offline |
| FPS calculation | ✅ Computed per window |
| Resource cleanup (`cap.release()`) | ✅ In `_capture_loop` |
| Bounded queue (no memory leak) | ✅ `maxsize=64` |
| Camera pre-check before starting | ✅ `cv2.VideoCapture(source).isOpened()` in `app.py` |
| Alert persistence and cooldown | ✅ `AlertManager` with `source_name` |
| **Hardware verification** | ⚠️ NOT TESTED — no webcam available during audit |

The code is correctly implemented. Hardware runtime verification remains outstanding.

---

## J. Dashboard Audit

**All 6 pages verified via `scripts/test_app_pages.py` (Streamlit AppTest):**

| Page | Status | Widgets rendered |
|------|--------|-----------------|
| Overview | ✅ PASS | 6 |
| Video Analysis | ✅ PASS | 1 |
| Live Monitor | ✅ PASS | 4 |
| Event Log | ✅ PASS | 2 |
| Model Performance | ✅ PASS | 9 |
| About | ✅ PASS | 1 |

**Dashboard connects to real model:** Confirmed — `@st.cache_resource` loads actual `.pkl` checkpoint.

**No fabricated metrics:** Evaluation page reads from `outputs/reports/evaluation_metrics.json` which was generated by real held-out predictions.

**Video upload workflow:** Guarded against re-processing on reruns via session-state hash. Temp file cleanup in all exit paths.

**Alert source naming:** Uses `source_name=uploaded.name` (filename), not numeric `"0"`.

**Known limitation:** Browser-level interaction testing (actual file upload click) was not performed — Streamlit's AppTest framework tests render logic only.

---

## K. Automated Testing Results

**Command:** `.\.venv\Scripts\python.exe -m pytest tests/ -v`

**Result: 59 passed, 0 failed, 0 warnings** (in 13.66s)

| Test file | Tests | Result |
|-----------|-------|--------|
| `test_alert_manager_constructor.py` | 8 | ✅ All pass |
| `test_alerts.py` | 14 | ✅ All pass |
| `test_evaluate.py` | 5 | ✅ All pass |
| `test_features.py` | 5 | ✅ All pass |
| `test_inference.py` | 3 | ✅ All pass |
| `test_inventory.py` | 4 | ✅ All pass |
| `test_model.py` | 6 | ✅ All pass |
| `test_split.py` | 4 | ✅ All pass |
| `test_video_analysis.py` | 10 | ✅ All pass |

Coverage areas: alert logic, threshold boundary, unique IDs, ack/dismiss, feature vector shape, corrupt video handling, data leakage, model save/load, split integrity, inference pipeline.

---

## L. Documentation and Reproducibility Audit

| Document | Status |
|----------|--------|
| `README.md` | ✅ Complete — pipeline, commands, limitations |
| `docs/USER_GUIDE.md` | ✅ Updated — uses venv commands |
| `docs/VIVA_PREPARATION.md` | ✅ Complete — 26 Q&A pairs |
| `docs/SYSTEM_ARCHITECTURE.md` | ✅ Complete — Mermaid diagram |
| `docs/TECHNICAL_DECISIONS.md` | ✅ Complete |
| `docs/CODEC_DIAGNOSIS.md` | ✅ Complete — root cause of non-decodable files |
| `docs/DATASET_REPORT.md` | ✅ Complete |
| `docs/PROJECT_ASSESSMENT.md` | ✅ Complete |
| `docs/IMPLEMENTATION_PLAN.md` | ✅ Complete |
| `docs/FINAL_REPORT.md` | ✅ Complete |

**Hardcoded absolute paths in source code:** 0 found  
**TODOs / NotImplementedError in source:** 0 found  
**Fake claims in documentation:** None detected — project consistently states prototype limitations

---

## M. Issues Found

### Critical Issues
None. The pipeline runs end-to-end without errors.

### Major Issues
1. **Dataset path mismatch** — `D:\7th sem\DL\Mini_Project\Datasets` does not exist. Videos are inside the project. The config correctly compensates but anyone following the original brief expecting `Datasets/` to exist will be confused. *Documenting this clearly is the only fix needed.*

2. **5-video test set** — Statistically insufficient for reliable evaluation. Balanced accuracy 0.583 ≈ chance. This is an honest result given the data available, not a bug, but faculty must be told this clearly.

3. **9 non-decodable videos** — Normal: 3, Extra: 6. These are permanently excluded. Could be recovered with FFmpeg (utility provided: `scripts/convert_unreadable_videos.py`), but FFmpeg is not installed.

### Minor Issues
4. **Webcam not hardware-tested** — The implementation is correct but was never run against a real camera.

5. **System Python 3.14 stale `.pyc` risk** — If someone runs `streamlit run app.py` with system Python instead of the venv, stale cached modules may cause `TypeError`. Fix: always use `.\.venv\Scripts\streamlit.exe run app.py`.

6. **Model performance is near-chance** — This is a dataset size issue, not a code bug. Documented honestly throughout.

### Optional Improvements
7. Collect more labelled videos (50+ per class minimum for meaningful evaluation).
8. Add temporal deep learning (LSTM/3D-CNN) with a larger dataset.
9. Add FFmpeg to the system PATH so non-decodable MOVs can be converted.

---

## N. Prioritised Remediation Plan

| Priority | Issue | Action |
|----------|-------|--------|
| P1 | Dataset path (`Datasets/`) absent | Clarify in README/viva that videos are inside `config/` and `dataset/` |
| P2 | Very small test set | State clearly in viva: "5 test videos is insufficient for statistical conclusions" |
| P3 | 9 non-decodable files | Optional: install FFmpeg, run `convert_unreadable_videos.py` |
| P4 | Webcam untested | Demonstrate webcam page in viva — note hardware verification was not automated |
| P5 | System Python stale cache risk | Always launch with venv command (already documented in USER_GUIDE.md) |

---

## O. Commands to Reproduce Verified Tests

```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"

# Run all 59 automated tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run end-to-end video analysis verification (10 steps)
.\.venv\Scripts\python.exe scripts/debug_video_analysis.py

# Run inference on test split
.\.venv\Scripts\python.exe scripts/quick_infer.py

# Test all 6 dashboard pages
.\.venv\Scripts\python.exe scripts/test_app_pages.py

# Launch the dashboard
.\.venv\Scripts\streamlit.exe run app.py
```

---

## P. Acceptance Criteria Checklist

| Criterion | Result |
|-----------|--------|
| Real dataset discovered correctly | ✅ PASS — 43 videos, correct labels |
| Original video files unchanged | ✅ PASS — read-only audit |
| Inventory report matches files on disk | ✅ PASS — 43 rows in manifest |
| Preprocessing executes correctly | ✅ PASS — features_train/val/test.npz exist, shape (9,33)/(3,33)/(5,33) |
| Baseline model trained on labelled data | ✅ PASS — 9 train videos, RF classifier |
| Training and evaluation data separated | ✅ PASS — 0 leakage videos |
| Evaluation metrics from actual held-out predictions | ✅ PASS — confusion matrix verified |
| Saved model reloads correctly | ✅ PASS — Pipeline, has predict_proba |
| Application analyses an uploaded video | ✅ PASS — debug script confirms 10/10 steps |
| Results from actual model inference | ✅ PASS — no fabricated outputs |
| Invalid inputs produce understandable errors | ✅ PASS — tested in test_video_analysis.py |
| Dashboard starts successfully | ✅ PASS — HTTP 200 confirmed |
| Real-time functionality tested | ⚠️ PARTIAL — code verified, hardware not tested |
| Tests and documentation reflect actual behaviour | ✅ PASS |
| No fabricated accuracy figures or alerts | ✅ PASS |
| 59 automated tests pass | ✅ PASS |
| All 6 dashboard pages render without exceptions | ✅ PASS |
