# Viva Preparation — Crowd Anomaly Detection

Likely questions from faculty and accurate, honest answers based on
what this project actually implements.

---

## Section 1: Project Understanding

**Q1: What is the problem you are solving?**
> We detect unusual motion patterns in crowd videos that may indicate abnormal
> crowd behaviour. The system outputs an anomaly score between 0 and 1.
> A score above the threshold triggers an alert for human review.
> It does NOT confirm emergencies — it is a decision-support prototype.

**Q2: What is the difference between normal and abnormal crowd behaviour?**
> Normal crowds show organised, low-speed, directionally consistent movement
> (e.g., people walking in a direction). Abnormal crowds show high-speed,
> chaotic, disorganised movement (e.g., people scattering, rushing, stampeding).
> The distinction is based on motion statistics, not visual content.

**Q3: Can this system detect panic?**
> No. The system classifies motion patterns as statistically similar to
> normal or abnormal training examples. It cannot infer psychological states
> like panic, fear, or intent. We explicitly do not claim this.

---

## Section 2: Dataset

**Q4: How many videos are in your dataset?**
> 43 total. 12 normal, 8 abnormal (labelled), and 23 extra unlabelled videos.
> Of the labelled 20, 17 are decodable by OpenCV. The 3 unreadable normal
> videos use a zlib-compressed QuickTime format unsupported by OpenCV on Windows.

**Q5: Why couldn't you use all the videos?**
> Nine .mov files produce the error "this file requires zlib support compiled in".
> OpenCV's bundled FFmpeg backend on Windows doesn't include zlib for this
> specific container format. We documented these and excluded them.

**Q6: Why didn't you label the extra videos?**
> The extra folder has no human-provided labels. Automatically assigning labels
> without domain expert review would introduce incorrect supervision and
> potentially harm model accuracy. We kept them unlabelled and separate.

**Q7: How did you split the dataset?**
> At the video level — all frames from the same video stay in one split.
> We used stratified splitting to maintain class proportions.
> Train: 9 videos (5 normal, 4 abnormal), Val: 3, Test: 5.
> This prevents data leakage that would occur with frame-level splitting.

---

## Section 3: Feature Engineering

**Q8: What features do you extract from videos?**
> 33 features per video, derived from motion analysis:
> - Dense optical flow (Farneback algorithm) between consecutive sampled frames
> - Flow magnitude mean, std, max — measures crowd speed
> - High-motion pixel fraction — what fraction of the frame is moving fast
> - Direction entropy — how disorganised the motion directions are (high in abnormal crowds)
> - Frame-difference mean, std, changed-pixel fraction — simpler change measure
> - These 8 per-pair stats are aggregated (mean, std, max, min) across all frame pairs → 32 values, plus a pair count → 33 total.

**Q9: What is optical flow?**
> Optical flow estimates the apparent motion of pixels between two consecutive
> frames. Dense optical flow computes a motion vector at every pixel.
> We use the Farneback algorithm (a polynomial expansion method) which is
> accurate enough and runs on CPU without a GPU.

**Q10: Why did you choose these features over deep learning?**
> Our dataset has only 17 labelled decodable videos. Any deep learning model
> would immediately overfit to the training set and fail to generalise.
> Hand-crafted motion features + Random Forest is the appropriate choice for
> a dataset this small. It also gives interpretable results.

---

## Section 4: Model

**Q11: What machine learning model do you use?**
> A scikit-learn Pipeline of: StandardScaler → RandomForestClassifier.
> The scaler normalises features. The Random Forest uses 100 decision trees
> with balanced class weights to handle mild class imbalance.

**Q12: Why Random Forest?**
> Handles small datasets well, outputs probabilities, is interpretable via
> feature importances, requires no GPU, trains in under 1 second on this data.

**Q13: What does class_weight='balanced' do?**
> It automatically adjusts tree weights so that the minority class (abnormal,
> fewer training examples) is not penalised. Without it, the model might
> default to always predicting "normal" to minimise error count.

**Q14: What is StandardScaler and why is it inside the Pipeline?**
> StandardScaler subtracts the mean and divides by standard deviation for each
> feature, bringing them all to a similar scale. Placing it inside the sklearn
> Pipeline ensures it is fitted ONLY on training data — if we fitted it on
> the full dataset first, the model would have seen test-set statistics during
> training, which is data leakage.

---

## Section 5: Evaluation

**Q15: What are your evaluation results?**
> On the 5-video held-out test set:
> - Balanced accuracy: 0.583
> - ROC-AUC: 0.500
> - TP=1, TN=2, FP=1, FN=1
> The model correctly classified 3 out of 5 test videos.

**Q16: Is a balanced accuracy of 0.583 good?**
> No, and we don't claim it is. Random chance gives 0.5 balanced accuracy.
> Our model is barely above chance. This is honest and expected:
> 9 training videos is an extremely small sample. The results demonstrate
> the pipeline works end-to-end, not that the model is deployment-ready.
> With hundreds of labelled videos the accuracy would be expected to improve.

**Q17: What is balanced accuracy and why use it instead of regular accuracy?**
> Balanced accuracy = (recall_normal + recall_abnormal) / 2. It gives equal
> weight to both classes regardless of how many samples each has. Regular
> accuracy would be misleading when classes are imbalanced.

**Q18: What is ROC-AUC?**
> Area under the Receiver Operating Characteristic curve. It measures how
> well the model ranks positives above negatives across all thresholds.
> 0.5 = random, 1.0 = perfect. Ours is 0.5 — the model's score ordering
> is not reliable, again reflecting the tiny dataset.

**Q19: Why is there data leakage risk in frame-level splitting?**
> If frames from the same video appear in both train and test splits,
> the model learns the specific video's appearance/motion style during training,
> then "predicts" that same style in testing. Performance looks great but it's
> memorisation, not generalisation.

---

## Section 6: System Design

**Q20: Explain the alert system.**
> Three conditions must all be true before an alert fires:
> 1. Anomaly score ≥ threshold (configurable, default 0.55)
> 2. Persists for N consecutive windows (default 3) — avoids single-frame noise
> 3. Cooldown period elapsed since last alert (default 10s) — avoids spam
> This design mirrors industrial alarm systems.

**Q21: What is a temporal window?**
> A window is a fixed number of consecutive sampled frames (e.g., 20 frames).
> Features are computed per window. This gives time-resolved scores —
> you can see when in the video the anomaly score rises rather than just
> getting a single label for the whole video.

**Q22: How does live inference work?**
> A background thread reads frames from the webcam or RTSP stream continuously.
> The main loop pulls frames from a bounded queue, builds windows of sampled
> frames, computes motion features, and calls the same model used for offline
> videos. The callback updates the dashboard display.

**Q23: How is the Streamlit dashboard connected to the real model?**
> The model is loaded once at app startup via `@st.cache_resource`. When a
> user uploads a video and clicks Analyse, `analyse_video()` processes it
> with the actual loaded sklearn pipeline. No scores are hardcoded.

---

## Section 7: Limitations and Ethics

**Q24: What are the main limitations?**
> 1. Extremely small dataset — results are not statistically reliable.
> 2. No temporal deep learning — sequence modelling could improve results.
> 3. Motion-only features — ignores crowd density, spatial patterns, scene context.
> 4. No cross-dataset validation — unknown how it performs on unseen camera angles or crowds.
> 5. Nine videos unreadable due to codec incompatibility.

**Q25: What ethical considerations apply?**
> - No face recognition, biometric identification, or demographic profiling.
> - The system must only support human decisions, not replace them.
> - False positives can cause unnecessary panic; false negatives can miss real events.
> - Privacy of individuals in footage must be protected.
> - Not suitable for deployment without extensive validation.

**Q26: How would you improve this project with more time/data?**
> - Use UCF-Crime or ShanghaiTech dataset (thousands of labelled clips).
> - Add temporal deep learning (LSTM over frame-level CNN features, or 3D-CNN).
> - Add crowd density estimation.
> - Cross-camera and cross-scene validation.
> - Uncertainty quantification so the model says "I don't know" for ambiguous inputs.

---

## Section 8: Quick Demo Script (5–7 minutes)

1. **[1 min]** Open `README.md` — explain the problem, dataset, approach.
2. **[1 min]** Run `python scripts/quick_infer.py` — show live predictions.
3. **[2 min]** `streamlit run app.py` — navigate to Overview (show charts),
   Video Analysis (upload one test video, click Analyse, show score gauge
   and timeline), Model Performance (show confusion matrix + metrics).
4. **[1 min]** Show `outputs/plots/confusion_matrix.png` and explain TP/FP/FN.
5. **[1 min]** `python -m pytest tests/ -v` — show 33 tests passing.
6. **[30 sec]** Summarise limitations honestly — shows academic integrity.

---

*This document reflects the actual implemented system. Do not claim results or
capabilities beyond what is shown in the evaluation outputs.*
