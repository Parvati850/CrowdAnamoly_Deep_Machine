# Project Assessment

Generated after Phase 1 inspection and Phase 6 evaluation.

---

## 1. Dataset Summary

| Category | Videos | Decodable | Used for training |
|----------|--------|-----------|-------------------|
| Normal Crowds | 12 | 9 | Yes (labelled) |
| Abnormal Crowds | 8 | 8 | Yes (labelled) |
| Crowd Dataset - extra | 23 | 17 | No (unlabelled) |
| **Total** | **43** | **34** | — |

### Non-decodable files (9)
OpenCV's Windows MSMF backend cannot decode these .mov files because they use
a zlib-compressed QuickTime container. Error: `this file requires zlib support
compiled in / error reading header`.

Files: `81872953_l.mov`, `sfw20110024_l.mov`, `shibuya1.mov` (Normal);
`1379-7_l.mov`, `1379-8_l.mov`, `81712424_l.mov`, `82409835_l.mov`,
`82409905_l.mov`, `965-83_l.mov` (Extra).

**FFmpeg fallback:** Not attempted — OpenCV's FFMPEG backend is also unable
to decode these because they require a zlib-enabled build. Installing
`ffmpeg` externally and using it via subprocess could decode them, but this
was not required since enough decodable videos exist.

---

## 2. Labelled Decodable Video Split

| Split | Normal | Abnormal | Total |
|-------|--------|----------|-------|
| Train | 5 | 4 | 9 |
| Val   | 1 | 2 | 3 |
| Test  | 3 | 2 | 5 |
| **Total** | **9** | **8** | **17** |

Splits are assigned at the video level (no frame-level leakage).
Stratified by label. Reproducible via `random_seed=42`.

---

## 3. Actual Evaluation Results (held-out test set, 5 videos)

| Metric | Value |
|--------|-------|
| Balanced Accuracy | 0.583 |
| ROC-AUC | 0.500 |
| True Positives (abnormal correctly caught) | 1 |
| True Negatives (normal correctly passed) | 2 |
| False Positives (normal flagged as abnormal) | 1 |
| False Negatives (abnormal missed) | 1 |

**Normal class:** Precision=0.667, Recall=0.667, F1=0.667
**Abnormal class:** Precision=0.500, Recall=0.500, F1=0.500

---

## 4. Limitations

- **Extremely small dataset.** 17 labelled decodable videos is far too few for
  any reliable generalisation. Results will vary significantly with different
  random seeds or slightly different splits.
- **Class imbalance.** Slightly more normal than abnormal (9 vs 8 decodable).
- **No temporal grouping.** Videos that may be from the same recording session
  were not explicitly grouped; this could introduce mild leakage if clips are
  related.
- **Single feature type.** Motion-only features (optical flow + frame diff)
  ignore spatial crowd structure, density estimation, and scene context.
- **Codec limitations.** 9/43 videos are unreadable by OpenCV on this system.

---

## 5. Recommended Baseline

**Random Forest on motion features (implemented).** Rationale:
- Lightweight, CPU-friendly.
- Interpretable feature importances.
- Handles small datasets better than deep models.
- No GPU required.

---

## 6. Recommended Development Order

1. ✅ Phase 1: Dataset inspection
2. ✅ Phase 2: Environment & dependency setup
3. ✅ Phase 3+4: Manifest, splits, feature extraction
4. ✅ Phase 5: Baseline training
5. ✅ Phase 6: Group-aware evaluation
6. ✅ Phase 7: Single-video inference
7. ✅ Phase 8: Streamlit dashboard
8. Phase 9: Webcam/RTSP integration (hardware-dependent)
9. Phase 10: Temporal deep-learning extension (needs larger dataset)
