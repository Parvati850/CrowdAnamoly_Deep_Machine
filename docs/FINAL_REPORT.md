# Final Report — Audit and Improvement

## 1. Original Issues Found

### Severity 1 — Bugs fixed

| # | Issue | Root cause | File | Fix |
|---|-------|-----------|------|-----|
| B1 | Video Analysis re-processed on every Streamlit rerun | No session-state guard on analysis result | `app.py` | Hash uploaded file; store result in `st.session_state`; skip if same hash |
| B2 | Temp file leaked when decode failed early | `tmp_path.unlink()` was only in the success branch | `app.py` | Moved cleanup to a `finally`-equivalent path covering error exits |
| B3 | OpenCV capture not released on exception | `cap.release()` was after the `while` loop, not in `finally` | `video_inference.py` | Wrapped frame-reading loop in `try/finally: cap.release()` |
| B4 | Event Log source showed `"0"` | `alert_manager.update(source=str(source))` where source is webcam int | `realtime.py`, `app.py` | Added `source_name` to `AlertManager` constructor; removed per-call source arg |
| B5 | Duplicate alert events on rerun | `st.session_state.alert_events.extend(alert_mgr.events)` inside analysis block without guard | `app.py` | New manager per analysis run; extend only once after run completes |
| B6 | No unique alert ID | Events had no identifier; ack/dismiss used index (breaks on reruns) | `alerts.py` | Added `event_id = uuid.uuid4()[:8]`; `acknowledge`/`dismiss` now take ID string |
| B7 | Score == threshold showed NORMAL in gauge but fired alert | Gauge used `score < threshold` but AlertManager used `score >= threshold` | `app.py`, `alerts.py` | Documented consistently: both use `>=`; sidebar caption added |
| B8 | Docstring said "34 values" for 33-value feature vector | Copy-paste error from earlier draft | `motion_features.py` | Corrected to 33 |

### Severity 2 — UX issues fixed

| # | Issue | Fix |
|---|-------|-----|
| U1 | No video metadata before analysis | Added FPS, dimensions, duration, frame count display after upload |
| U2 | No video playback preview | Added `st.video(uploaded)` in collapsible expander |
| U3 | Analyse button re-enabled after result shown causing re-runs | Button disabled when same-file result already exists |
| U4 | "Clear session alerts" implied it cleared disk log | Added explicit caption: "Persisted log on disk is unchanged" |
| U5 | Live Monitor: no camera availability check before starting | Added `cv2.VideoCapture` pre-check with troubleshooting steps |
| U6 | Live Monitor Start button clickable while running | Added `disabled=st.session_state.live_running` |
| U7 | Model Performance: ROC-AUC = 0.5 with no explanation | Added `st.warning` explaining near-chance performance is expected |
| U8 | Model Performance: train accuracy = 1.0 with no explanation | Added caption: "RF can memorise 9 samples — not a generalisation measure" |
| U9 | Per-video test breakdown not shown | Added expandable table on Model Performance page |

---

## 2. Root Causes (confirmed by evidence)

**Rerun duplication:** Streamlit re-executes `app.py` top-to-bottom on every
interaction. Without session-state guards, the analysis block ran repeatedly,
each run extending `alert_events` by the same events.

**Source showing "0":** `AlertManager.update()` was called with `source=str(source)`
where `source` is the integer `0` for webcam index 0. This produced the string
`"0"` as the source label in every event. Fixed by setting `source_name` once
at `AlertManager` construction and removing per-call overrides.

**cap not released:** The `while True` frame loop in `analyse_video` had
`cap.release()` after the loop. An exception inside `model_predict` would skip
it entirely, leaking the capture. Fixed with `try/finally`.

**Non-decodable MOVs:** zlib-compressed QuickTime container headers — not file
corruption, not wrong paths. The OpenCV bundled FFmpeg on Windows lacks zlib
support for this container variant. Documented in `docs/CODEC_DIAGNOSIS.md`.

---

## 3. Files Changed

| File | Changes |
|------|---------|
| `src/inference/alerts.py` | Added `event_id` (uuid), `source_name` constructor param, `acknowledge`/`dismiss` by ID, stable CSV field order |
| `src/inference/video_inference.py` | `cap.release()` in `try/finally`; removed `source=` from per-window alert calls |
| `src/inference/realtime.py` | Removed `source=str(source)` from alert update call |
| `src/features/motion_features.py` | Fixed docstring: "33 values", not "34" |
| `app.py` | Complete rewrite of all 6 pages with all fixes above |
| `tests/test_alerts.py` | Rewritten: 14 tests covering threshold boundary, unique IDs, source_name, ack/dismiss, CSV fields |
| `tests/test_video_analysis.py` | New file: 10 tests for corrupt video, dedup, threshold equality, cap release |
| `docs/CODEC_DIAGNOSIS.md` | New: root-cause analysis of non-decodable files |
| `docs/FINAL_REPORT.md` | This file |
| `scripts/convert_unreadable_videos.py` | New: optional FFmpeg conversion utility |
| `README.md` | Updated: pipeline explanation, changes summary, test count |

---

## 4. Commands to Run the Project

```powershell
cd "D:\7th sem\DL\Mini_Project\CrowdAnomalyDetection"

# Full pipeline (first time or after clearing generated data)
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python scripts/train_model.py
python scripts/evaluate_model.py

# Spot-check inference
python scripts/quick_infer.py

# All automated tests
python -m pytest tests/ -v

# Dashboard
python -m streamlit run app.py
```

---

## 5. Tests Executed — Actual Results

```
51 passed, 0 failed, 1 warning (sklearn single-class UserWarning — cosmetic)
Run time: ~10 seconds
```

**Test breakdown:**
- `test_alerts.py` — 14 tests: threshold, persistence, cooldown, unique IDs, source_name, ack/dismiss, CSV, dedup
- `test_video_analysis.py` — 10 tests: feature vector length, corrupt video, predict valid/corrupt/missing, analyse dedup, cap release, score range, threshold boundary
- `test_evaluate.py` — 5 tests: metrics correctness
- `test_features.py` — 5 tests: vector shape, NaN, names count, invalid video, short video
- `test_inference.py` — 3 tests: predict valid, predict missing, analyse windows
- `test_inventory.py` — 4 tests: discovery, unreadable, extra label, save/reload
- `test_model.py` — 6 tests: train, predict shape, save/load, missing checkpoint, empty train, classifiers
- `test_split.py` — 4 tests: all labelled split, no leakage, extra not split, stratification

Dashboard page tests (AppTest): **6/6 pages pass, 0 exceptions**

---

## 6. How to Demonstrate Video Analysis in a College Review

**5-minute demo sequence:**

1. Start dashboard: `python -m streamlit run app.py`
2. Navigate to **Video Analysis** page
3. Upload one of the real test videos from `config/Abnormal Crowds/2017-420_l.mov`
4. Show the metadata panel (FPS, dimensions, duration) and preview
5. Click **▶ Analyse Video** — show the spinner
6. When complete, point out:
   - The score gauge (0.56 → above threshold → ANOMALY DETECTED)
   - The "This is a model prediction — not a confirmed emergency" warning
   - The window score timeline (if windows were produced)
7. Navigate to **Model Performance** — show the confusion matrix and honest metrics
8. Show the `st.warning` about the test set being only 5 videos
9. Run `python -m pytest tests/ -v` in the terminal to show 51 passing tests
10. Navigate to **About** — explain the pipeline step by step

---

## 7. Remaining Limitations

1. **17 labelled decodable videos** — too few for reliable generalisation.
   Balanced accuracy 0.583, ROC-AUC 0.50 (near chance).
2. **9 MOV files unreadable** — codec limitation, not data corruption.
   Recovery possible with external FFmpeg.
3. **No temporal deep learning** — LSTM/3D-CNN deferred (needs larger dataset).
4. **Live monitor not camera-tested** — webcam hardware was not available during
   development. The code path is implemented and pre-checked but hardware
   validation was not performed.
5. **Single feature modality** — motion only. No crowd density, spatial structure,
   or scene context.

---

## 8. Pipeline Explanation (beginner-friendly)

```
VIDEO INPUT
  The user uploads a crowd video (AVI, MOV, MP4, etc.)

FRAME DECODING
  OpenCV reads the video frame by frame.
  We keep every 10th frame to reduce computation.
  Each kept frame is resized to 224×224 pixels.

MOTION FEATURE EXTRACTION
  For each pair of consecutive kept frames:
  • Dense Optical Flow (Farneback algorithm) estimates how every pixel moved.
    → We measure: average speed, speed spread, max speed,
                  fraction of fast-moving pixels,
                  direction entropy (how chaotic the movement is).
  • Frame Difference: raw pixel-level change between frames.
    → We measure: mean change, spread, fraction of changed pixels.
  These 8 measurements are aggregated (mean, std, max, min) across all pairs
  → 32 values + 1 pair count = 33 numbers total.

MODEL PREDICTION
  A StandardScaler normalises the 33 numbers to a common scale.
  A Random Forest (100 decision trees) trained on 9 crowd videos
  outputs two probabilities: P(normal) and P(abnormal).
  These sum to 1.0.

THRESHOLD DECISION
  If P(abnormal) >= threshold (default 0.55): the window is "suspicious".
  If this persists for 3 consecutive windows AND the cooldown has elapsed:
  an ALERT fires. Each alert gets a unique ID and is logged.

HUMAN REVIEW
  The dashboard shows the score, the alert, and an explicit warning:
  "This is a model prediction — not a confirmed emergency."
  A human operator decides whether any action is warranted.
```

---

## 9. Recommended Future Work

Priority order:

1. **Collect more data.** Minimum 100+ labelled videos per class for meaningful evaluation.
2. **Add temporal modelling.** Replace per-window RF with an LSTM over frame-level CNN features.
3. **Crowd density estimation.** Combine motion with density to distinguish crowded-but-calm from sparse-but-chaotic.
4. **Cross-dataset validation.** Test on UCF-Crime or ShanghaiTech to measure generalisation.
5. **Convert unreadable MOVs.** Use the provided `convert_unreadable_videos.py` with FFmpeg to recover 9 files.
6. **Uncertainty output.** Report confidence intervals so the model can say "I'm not sure" on ambiguous clips.
