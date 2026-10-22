"""
app.py
------
CrowdAnomalyDetection – Streamlit Monitoring Dashboard

Launch:
    cd "D:\\7th sem\\DL\\Mini_Project\\CrowdAnomalyDetection"
    python -m streamlit run app.py

All predictions and scores come from the real trained model.
Nothing is fabricated.
"""

import sys
import json
import tempfile
import time
import hashlib
from pathlib import Path

import cv2
import streamlit as st

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_config, get_path
from src.utils.logging_config import setup_logging
from src.dashboard.components import (
    model_status_banner, score_gauge, window_score_chart,
    event_log_table, metrics_display, confusion_matrix_image,
    dataset_summary_table,
)
from src.dashboard.visualizations import (
    label_distribution_chart, decodability_chart,
    score_history_chart, roc_curve_image,
)

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crowd Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached loaders (run once per server process) ─────────────────────────────
@st.cache_resource
def _load_cfg():
    return load_config()

cfg = _load_cfg()
setup_logging(cfg["logging"]["level"])

MANIFEST_PATH = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
REPORTS_DIR   = get_path(cfg, "output", "reports_dir")
PLOTS_DIR     = get_path(cfg, "output", "plots_dir")
CKPT_PATH     = (get_path(cfg, "training", "model_output_dir")
                 / cfg["training"]["checkpoint_name"])

@st.cache_resource
def _load_model():
    if not CKPT_PATH.exists():
        return None, False
    try:
        from src.models.baseline import load_model
        return load_model(CKPT_PATH), True
    except Exception:
        return None, False

model, model_loaded = _load_model()

# ── Deep model loader ─────────────────────────────────────────────────────────
DEEP_CKPT_PATH = (get_path(cfg, "training", "model_output_dir")
                  / cfg["training"].get("deep_checkpoint_name", "deep_model.pth"))

@st.cache_resource
def _load_deep_model():
    if not DEEP_CKPT_PATH.exists():
        return None, False
    try:
        from src.models.deep_model import load_deep_model
        return load_deep_model(DEEP_CKPT_PATH), True
    except Exception:
        return None, False

deep_model, deep_model_loaded = _load_deep_model()

@st.cache_data
def _load_metrics():
    p = REPORTS_DIR / "evaluation_metrics.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}

eval_metrics = _load_metrics()

@st.cache_data
def _load_deep_metrics():
    p = REPORTS_DIR / "evaluation_metrics_deep.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}

eval_metrics_deep = _load_deep_metrics()

# ── Session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("alert_events",        []),
    ("last_video_hash",     None),
    ("last_analysis_result", None),
    ("last_video_name",     None),
    ("live_running",        False),
    ("live_scores",         []),
    ("live_analyser",       None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Helper: stable fingerprint of an uploaded file ───────────────────────────
def _file_hash(uploaded_file) -> str:
    """Return a short MD5 of the first 64 KB for fast dedup."""
    data = uploaded_file.read(65536)
    uploaded_file.seek(0)
    return hashlib.md5(data).hexdigest()


# ── Helper: probe video metadata without modifying the file ──────────────────
def _probe_video(path: Path) -> dict:
    """Return fps, width, height, frame_count, duration_s, or None on failure."""
    info = {"fps": None, "width": None, "height": None,
            "frame_count": None, "duration_s": None, "decodable": False}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return info
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"]         = round(fps, 2) if fps > 0 else None
        info["width"]       = w if w > 0 else None
        info["height"]      = h if h > 0 else None
        info["frame_count"] = fc if fc > 0 else None
        if info["fps"] and info["frame_count"]:
            info["duration_s"] = round(info["frame_count"] / info["fps"], 1)
        ret, frame = cap.read()
        info["decodable"] = bool(ret and frame is not None)
    finally:
        cap.release()
    return info


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/crowd.png", width=80)
    st.title("Crowd Anomaly Detection")
    st.caption("Decision-Support Prototype · Not for production use")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Video Analysis", "Live Monitor", "Event Log",
         "Model Performance", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    threshold = st.slider(
        "Alert Threshold",
        min_value=0.10, max_value=0.95,
        value=float(cfg["alerts"]["threshold"]),
        step=0.05,
        help=(
            "Score ≥ this value is treated as suspicious.  "
            "An alert fires after this persists for "
            f"{cfg['alerts']['persistence']} consecutive windows."
        ),
    )
    st.caption(f"Score **≥ {threshold:.2f}** triggers alert (equal-to included).")

    st.divider()
    # ── Model selector ────────────────────────────────────────────────────
    _model_options = ["Random Forest (baseline)"]
    if deep_model_loaded:
        _model_options.append("CNN-LSTM (experimental ⚠)")
    else:
        _model_options.append("CNN-LSTM — not trained yet")

    selected_model_label = st.radio(
        "Analysis model",
        _model_options,
        index=0,
        help=(
            "Random Forest uses 33 handcrafted motion features.  "
            "CNN-LSTM uses a frozen MobileNetV3 frame encoder + LSTM.  "
            "CNN-LSTM is experimental: trained on only 11 videos."
        ),
    )
    use_deep = selected_model_label.startswith("CNN-LSTM") and deep_model_loaded

    model_status_banner(model_loaded, str(CKPT_PATH))


# ════════════════════════════════════════════════════════════════════════════
# Page: Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🚨 Crowd Anomaly Detection Dashboard")
    st.markdown(
        """
        This system analyses crowd video footage for unusual motion patterns.

        > **Limitation:** This is a decision-support prototype. It detects
        > potentially unusual motion — not confirmed panic, injury, or emergencies.
        > Every alert requires human review before any action is taken.
        """
    )
    st.divider()

    # ── Dataset statistics ────────────────────────────────────────────────
    import pandas as pd
    st.subheader("Dataset")
    if MANIFEST_PATH.exists():
        mdf = pd.read_csv(MANIFEST_PATH)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Videos",    len(mdf))
        c2.metric("Normal (labelled)",   int((mdf["label"] == "normal").sum()))
        c3.metric("Abnormal (labelled)", int((mdf["label"] == "abnormal").sum()))
        c4.metric("Unlabelled (extra)",  int((mdf["label"] == "unknown").sum()))
        c5.metric("Non-decodable",       int((~mdf["decoded"]).sum()))
    else:
        st.warning("Dataset manifest not found. Run `python scripts/inspect_dataset.py`.")

    # ── Charts ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(label_distribution_chart(MANIFEST_PATH), width="stretch")
    with col_b:
        st.plotly_chart(decodability_chart(MANIFEST_PATH), width="stretch")

    st.divider()

    # ── System status ─────────────────────────────────────────────────────
    st.subheader("System Status")
    if model_loaded:
        import json as _json
        tm_path = REPORTS_DIR / "train_metadata.json"
        if tm_path.exists():
            tm = _json.loads(tm_path.read_text())
            st.success(
                f"✅ Model ready  ·  {tm.get('classifier_type','?')} classifier  ·  "
                f"trained on {tm.get('n_train','?')} videos"
            )
        else:
            st.success("✅ Model checkpoint loaded.")
    else:
        st.error("❌ No trained model. Run the pipeline first:")
        st.code(
            "python scripts/inspect_dataset.py\n"
            "python scripts/prepare_dataset.py\n"
            "python scripts/train_model.py\n"
            "python scripts/evaluate_model.py"
        )

    st.divider()
    st.caption(
        "Non-decodable .mov files use a zlib-compressed QuickTime container "
        "not supported by OpenCV on Windows. They are excluded from training."
    )


# ════════════════════════════════════════════════════════════════════════════
# Page: Video Analysis
# ════════════════════════════════════════════════════════════════════════════
elif page == "Video Analysis":
    st.title("📹 Video Analysis")

    if not model_loaded:
        st.error(
            "No trained model found.  "
            "Run `python scripts/train_model.py` before using this page."
        )
        st.stop()

    # ── Upload area ───────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a crowd video for analysis",
        type=["avi", "mov", "mp4", "wmv", "flv", "mkv"],
        help="Supported: AVI, MOV, MP4, WMV, FLV, MKV.  "
             "MOV files with zlib-compressed QuickTime headers may not decode.",
    )

    if uploaded is None:
        st.info(
            "Upload a video file above to begin.  \n"
            "The model will extract motion features and estimate whether "
            "the crowd behaviour appears normal or unusual."
        )
        # Clear any stale result from a previous upload
        st.session_state.last_video_hash     = None
        st.session_state.last_analysis_result = None
        st.session_state.last_video_name     = None
        st.stop()

    # ── Video metadata (shown immediately after upload) ───────────────────
    current_hash = _file_hash(uploaded)
    same_file    = (current_hash == st.session_state.last_video_hash)

    file_size_kb = uploaded.size / 1024
    st.markdown(f"**File:** `{uploaded.name}`  · **Size:** {file_size_kb:.1f} KB")

    # Write to temp file once, then probe for metadata
    suffix = Path(uploaded.name).suffix
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            uploaded.seek(0)
            tmp_path = Path(tmp.name)

        meta = _probe_video(tmp_path)
    except Exception as e:
        st.error(f"Could not write temporary file: {e}")
        st.stop()

    # Show metadata
    if meta["decodable"]:
        mcols = st.columns(5)
        mcols[0].metric("FPS",    meta["fps"] or "?")
        mcols[1].metric("Width",  meta["width"] or "?")
        mcols[2].metric("Height", meta["height"] or "?")
        mcols[3].metric("Frames", meta["frame_count"] or "?")
        mcols[4].metric("Duration", f"{meta['duration_s']} s" if meta["duration_s"] else "?")

        # Playback preview (Streamlit can render common formats)
        with st.expander("▶ Preview uploaded video", expanded=False):
            st.video(uploaded)
    else:
        st.error(
            "❌ This video could not be decoded.  \n"
            "Possible causes: unsupported codec, zlib-compressed MOV container, "
            "or corrupted file.  \n"
            "Try converting to H.264 .mp4 or standard .avi (MJPEG)."
        )
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        st.stop()

    st.divider()

    # ── Analyse button — guarded against re-processing same file ─────────
    analyse_btn = st.button("▶ Analyse Video", type="primary",
                            disabled=(same_file and st.session_state.last_analysis_result is not None))

    if same_file and st.session_state.last_analysis_result is not None:
        st.info("Results below are from the most recent analysis of this file. "
                "Upload a different video or re-run to analyse again.")
    elif analyse_btn:
        if use_deep and not deep_model_loaded:
            st.error("CNN-LSTM checkpoint not found. Run `scripts/train_deep_model.py` first.")
            st.stop()

        with st.spinner(f"Analysing `{uploaded.name}` — please wait…"):
            from src.inference.alerts import AlertManager
            alert_mgr = AlertManager(
                threshold=threshold,
                persistence=cfg["alerts"]["persistence"],
                cooldown_seconds=cfg["alerts"]["cooldown_seconds"],
                source_name=uploaded.name,
            )

            if use_deep:
                # ── CNN-LSTM path ─────────────────────────────────────────
                from src.models.deep_predict import predict_video_deep
                _deep_result = predict_video_deep(tmp_path, cfg=cfg, model=deep_model)
                if _deep_result["error"]:
                    result = {"error": _deep_result["error"], "windows": [],
                              "overall_score": None, "overall_label": "unknown",
                              "fps_processing": 0.0, "alerts_fired": 0}
                else:
                    # Wrap single-video prediction into the windowed-result schema
                    # so the display code below works unchanged
                    _score = _deep_result["score_abnormal"]
                    alert_mgr.update(_score, _deep_result["label"])
                    result = {
                        "error":          None,
                        "windows":        [],     # CNN-LSTM gives a single video-level score
                        "overall_score":  _score,
                        "overall_label":  _deep_result["label"],
                        "fps_processing": 0.0,
                        "alerts_fired":   alert_mgr.active_count,
                        "model_type":     "cnn_lstm",
                    }
            else:
                # ── Random Forest path (unchanged) ────────────────────────
                from src.inference.video_inference import analyse_video
                result = analyse_video(
                    video_path=tmp_path,
                    cfg=cfg,
                    model=model,
                    alert_manager=alert_mgr,
                    window_size=20,
                )
                if not result.get("error"):
                    result["model_type"] = "random_forest"

        # Temp file is no longer needed — clean up
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        if result["error"]:
            st.error(f"Analysis failed: {result['error']}")
        else:
            # Store result to avoid re-processing on Streamlit reruns
            st.session_state.last_video_hash      = current_hash
            st.session_state.last_analysis_result = result
            st.session_state.last_video_name      = uploaded.name
            # Append only the NEW alerts from this analysis run
            st.session_state.alert_events.extend(alert_mgr.events)

    # ── Display results (from session state — survives reruns) ────────────
    import pandas as pd
    result = st.session_state.last_analysis_result
    if result is not None and not result.get("error"):
        st.subheader("Analysis Results")
        _mt = result.get("model_type", "random_forest")
        if _mt == "cnn_lstm":
            st.info(
                "⚗ Results from **CNN-LSTM (experimental)** — "
                "MobileNetV3-Small + LSTM, trained on 11 videos.  "
                "These results are not statistically reliable. "
                "Human review is required."
            )
        else:
            st.caption("Model: **Random Forest** (33 motion features)")

        overall_score = result["overall_score"] or 0.0
        col1, col2, col3 = st.columns(3)
        with col1:
            score_gauge(overall_score, threshold)
        with col2:
            st.metric("Windows Analysed", len(result["windows"]))
            st.metric("Processing FPS",   f"{result['fps_processing']:.1f}")
        with col3:
            st.metric("Alerts Fired", result["alerts_fired"])
            if result["overall_label"] == "abnormal":
                st.error(
                    "⚠ Unusual motion detected.  \n"
                    "This is a model prediction — not a confirmed emergency.  \n"
                    "Human review required."
                )
            else:
                st.success("✅ Motion appears within normal range for this model.")

        n_wins = len(result["windows"])
        n_abn  = sum(1 for w in result["windows"] if w["label"] == "abnormal")
        st.caption(
            f"Summary: {n_wins} window(s) analysed, "
            f"{n_abn} flagged above threshold ({threshold:.2f}).  "
            f"Frames sampled every {cfg['preprocessing']['frame_sample_interval']} frames.  "
            f"Score ≥ threshold triggers alert after "
            f"{cfg['alerts']['persistence']} consecutive windows."
        )

        if result["windows"]:
            st.divider()
            window_score_chart(result["windows"])

            with st.expander("Raw window data"):
                st.dataframe(
                    pd.DataFrame(result["windows"]),
                    width="stretch",
                    hide_index=True,
                )

        # ── Motion indicator explanation ──────────────────────────────────
        st.divider()
        st.subheader("What the score means")
        _mt_display = result.get("model_type", "random_forest")
        if _mt_display == "cnn_lstm":
            st.markdown(
                f"""
                The **score** (P(abnormal) = **{overall_score:.3f}**) is the
                CNN-LSTM's softmax output for the "abnormal" class.

                **How it was computed:** the video was sampled every
                {cfg['preprocessing']['frame_sample_interval']} frames, each frame
                was passed through a frozen MobileNetV3-Small backbone (576-dim features),
                and the resulting sequence was classified by a single-layer LSTM.

                This is **not** a calibrated probability of panic or danger. The
                model was trained on only 11 videos and predicts "normal" for all
                6 test videos in the current evaluation.  Treat this score as
                experimental only.
                """
            )
        else:
            st.markdown(
                f"""
                The **anomaly score** (P(abnormal) = **{overall_score:.3f}**) is the
                Random Forest's vote-fraction for the "abnormal" class across 100 trees.

                This is **not** a calibrated probability of panic or danger.
                It reflects how similar the video's 33 motion features are to the
                training abnormal videos versus training normal videos.

                **Motion features used:**
                - Optical flow magnitude (mean / std / max) — crowd movement speed
                - Direction entropy — how disorganised/chaotic the motion is
                - High-motion pixel fraction — proportion of the frame moving fast
                - Frame-difference statistics — raw pixel-level change

                With only 11 training videos these features may not generalise
                to unseen camera angles, crowd densities, or compression quality.
                """
            )

        # ── Model comparison reference ─────────────────────────────────────
        comp_path = REPORTS_DIR / "model_comparison.json"
        if comp_path.exists():
            with open(comp_path) as f:
                comp = json.load(f)
            with st.expander("Fair model comparison (same features, same split)"):
                comp_df = pd.DataFrame(comp["models"]).T
                st.dataframe(
                    comp_df[["val_balanced_accuracy", "test_balanced_accuracy",
                            "test_roc_auc", "tp", "tn", "fp", "fn"]],
                    use_container_width=True,
                )
                st.caption(
                    "DummyClassifier (random) sets the baseline. "
                    "If the trained model doesn't beat it, the model hasn't "
                    "learned generalisable patterns from this dataset."
                )

        if st.button("🔄 Analyse a Different Video"):
            st.session_state.last_video_hash      = None
            st.session_state.last_analysis_result = None
            st.session_state.last_video_name      = None
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# Page: Live Monitor
# ════════════════════════════════════════════════════════════════════════════
elif page == "Live Monitor":
    st.title("📡 Live Monitor")

    if not model_loaded:
        st.error("No trained model found. Run `python scripts/train_model.py` first.")
        st.stop()

    st.info(
        "Scores are generated by the real trained model from live motion features.  \n"
        "The browser cannot display a raw camera stream — only scores are shown here."
    )

    source_type = st.radio("Input source", ["Webcam", "RTSP URL"])
    if source_type == "Webcam":
        cam_idx = st.number_input("Webcam index", value=int(cfg["live"]["webcam_index"]),
                                  min_value=0, step=1)
        source = int(cam_idx)
        source_label = f"Webcam {cam_idx}"
    else:
        rtsp_url = st.text_input(
            "RTSP URL", value=cfg["live"].get("rtsp_url", ""),
            placeholder="rtsp://user:pass@192.168.1.1/stream",
        )
        if not rtsp_url:
            st.warning("Enter an RTSP URL to proceed.")
            st.stop()
        source = rtsp_url
        source_label = rtsp_url

    col_start, col_stop = st.columns(2)
    start_btn = col_start.button("▶ Start", type="primary",
                                 disabled=st.session_state.live_running)
    stop_btn  = col_stop.button("⏹ Stop",
                                disabled=not st.session_state.live_running)

    status_box     = st.empty()
    score_ph       = st.empty()
    chart_ph       = st.empty()
    fps_ph         = st.empty()

    if st.session_state.live_running:
        status_box.success(f"🟢 Monitoring: `{source_label}`")
    else:
        status_box.info("⚪ Not started. Press ▶ Start to begin.")

    if stop_btn and st.session_state.live_running:
        analyser = st.session_state.live_analyser
        if analyser is not None:
            analyser.stop()
        st.session_state.live_running  = False
        st.session_state.live_analyser = None
        status_box.warning("⏹ Monitoring stopped.")

    if start_btn and not st.session_state.live_running:
        # Quick check: can we actually open the source?
        test_cap = cv2.VideoCapture(source)
        if not test_cap.isOpened():
            test_cap.release()
            st.error(
                f"Cannot open source `{source_label}`.  \n"
                "**Troubleshooting:**  \n"
                "- Check the webcam index (try 0, 1, or 2)  \n"
                "- Ensure no other app is using the camera  \n"
                "- Verify RTSP URL and network connectivity  \n"
                "- Check camera drivers are installed"
            )
            st.stop()
        test_cap.release()

        st.session_state.live_running = True
        st.session_state.live_scores  = []
        status_box.success(f"🟢 Connecting to `{source_label}`…")

        from src.inference.alerts import AlertManager
        alert_mgr = AlertManager(
            threshold=threshold,
            persistence=cfg["alerts"]["persistence"],
            cooldown_seconds=cfg["alerts"]["cooldown_seconds"],
            source_name=source_label,
        )

        def on_result(score, label, fps):
            st.session_state["live_scores"].append(score)
            new_events = alert_mgr.events[
                len(st.session_state.alert_events):]
            st.session_state.alert_events.extend(new_events)
            scores = st.session_state["live_scores"][-60:]
            with score_ph.container():
                score_gauge(score, threshold)
            with chart_ph.container():
                fig = score_history_chart(scores, threshold)
                st.plotly_chart(fig, width="stretch")
            fps_ph.caption(
                f"Processing FPS: {fps}  ·  Windows processed: {len(scores)}  "
                f"·  Source: `{source_label}`  ·  {time.strftime('%H:%M:%S')}"
            )

        from src.inference.realtime import RealtimeAnalyser
        analyser = RealtimeAnalyser(
            cfg=cfg, model=model, alert_manager=alert_mgr,
            on_result=on_result, window_size=20,
        )
        st.session_state.live_analyser = analyser
        try:
            analyser.start(source)
        except Exception as exc:
            st.error(f"Live monitoring error: {exc}")
            st.session_state.live_running = False

    st.divider()
    st.caption(
        "⚠ Live inference runs in the Streamlit server process.  "
        "Performance depends on available CPU resources.  "
        "For production use, run inference in a separate process."
    )


# ════════════════════════════════════════════════════════════════════════════
# Page: Event Log
# ════════════════════════════════════════════════════════════════════════════
elif page == "Event Log":
    import pandas as pd
    st.title("📋 Event Log")

    # ── Persisted log (on-disk CSV) ───────────────────────────────────────
    alert_log_path = get_path(cfg, "alerts", "log_path")
    st.subheader("Persisted Alerts (saved to disk)")
    if alert_log_path.exists():
        log_df = pd.read_csv(alert_log_path)
        if log_df.empty:
            st.info("Alert log file exists but contains no events.")
        else:
            st.metric("Total persisted alerts", len(log_df))
            st.dataframe(log_df, width="stretch", hide_index=True)
    else:
        st.info(
            f"No alert log file found at `{alert_log_path}`.  "
            "Alerts are written here when analysis fires them."
        )

    st.divider()

    # ── Session log ───────────────────────────────────────────────────────
    st.subheader("Session Alerts (this browser session only)")
    st.caption(
        "Session alerts are held in memory. They are lost when you close the "
        "browser tab. They do NOT duplicate the persisted alerts above."
    )

    events = st.session_state.alert_events
    if not events:
        st.info("No alerts fired in this session.")
    else:
        st.metric("Session alerts", len(events))
        # Build display table with ack/dismiss controls
        rows = []
        for i, e in enumerate(events):
            rows.append({
                "ID":          e.event_id,
                "Time":        e.timestamp,
                "Source":      e.source,
                "Score":       f"{e.score_abnormal:.3f}",
                "Threshold":   f"{e.threshold:.2f}",
                "Label":       e.label,
                "Ack":         "✓" if e.acknowledged else "",
                "Dismissed":   "✓" if e.dismissed else "",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Per-event actions
        with st.expander("Acknowledge / Dismiss an event"):
            ev_ids = [e.event_id for e in events]
            chosen_id = st.selectbox("Select event ID", ev_ids)
            ca, cd = st.columns(2)
            if ca.button("✓ Acknowledge"):
                for e in st.session_state.alert_events:
                    if e.event_id == chosen_id:
                        e.acknowledged = True
                st.rerun()
            if cd.button("✗ Dismiss"):
                for e in st.session_state.alert_events:
                    if e.event_id == chosen_id:
                        e.dismissed = True
                st.rerun()

        if st.button("🗑 Clear session alerts"):
            st.session_state.alert_events = []
            st.caption("Session alerts cleared. Persisted log on disk is unchanged.")
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# Page: Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("📊 Model Performance")

    if not model_loaded:
        st.error("No trained model found. Run `python scripts/train_model.py` first.")
        st.stop()

    # ── Per-page model selector (matches sidebar choice) ─────────────────
    import pandas as pd
    _perf_options = ["Random Forest (baseline)"]
    if deep_model_loaded:
        _perf_options.append("CNN-LSTM (experimental ⚠)")
    perf_model_label = st.radio(
        "Show metrics for",
        _perf_options,
        index=0,
        horizontal=True,
        key="perf_model_radio",
    )
    _show_deep = perf_model_label.startswith("CNN-LSTM") and deep_model_loaded
    _active_metrics = eval_metrics_deep if _show_deep else eval_metrics

    st.divider()

    # ── Training info (RF only — deep training info in separate expander) ─
    if not _show_deep:
        train_meta_path = REPORTS_DIR / "train_metadata.json"
        if train_meta_path.exists():
            with open(train_meta_path, encoding="utf-8") as fh:
                tm = json.load(fh)
            st.subheader("Random Forest — Training Configuration")
            c1, c2, c3 = st.columns(3)
            c1.metric("Classifier",       tm.get("classifier_type", "?"))
            c2.metric("Training videos",   tm.get("n_train", "?"))
            c3.metric("Validation videos", tm.get("n_val", "?"))
            st.caption(
                f"Train-set accuracy = **{tm.get('train_accuracy_on_trainset','?')}** "
                f"— this is on the training data itself. A Random Forest with "
                f"{tm.get('n_train','?')} training videos can memorise them perfectly. "
                "This is NOT a measure of generalisation."
            )
            st.divider()
    else:
        deep_hist_path = REPORTS_DIR / "deep_train_history.json"
        st.subheader("CNN-LSTM — Training Summary")
        st.warning(
            "⚗ This model is **experimental**. It was trained on 11 videos "
            "with a frozen MobileNetV3-Small backbone + LSTM head. "
            "Results on a 6-video test set are not statistically reliable.",
            icon="⚗",
        )
        if deep_hist_path.exists():
            with open(deep_hist_path) as _f:
                _hist = json.load(_f)
            _ep = _hist.get("epochs", [])
            if _ep:
                c1, c2 = st.columns(2)
                c1.metric("Epochs trained", max(_ep))
                c2.metric("Best val loss",
                          min(_hist["val_loss"]) if _hist.get("val_loss") else "?")
        else:
            st.info("Training history not found. Run `scripts/train_deep_model.py`.")
        st.divider()

    # ── Test-set evaluation ───────────────────────────────────────────────
    if _active_metrics:
        n_test = len(_active_metrics.get("test_filenames", []))
        _model_label = "CNN-LSTM" if _show_deep else "Random Forest"
        st.subheader(f"{_model_label} — Held-out Test Set Evaluation ({n_test} videos)")

        st.warning(
            f"⚠ The test set contains only **{n_test} videos**.  "
            "Metrics from this few samples are statistically unreliable — "
            f"a single misclassification shifts accuracy by {100/n_test:.1f} pp.  "
            "These results demonstrate the pipeline works end-to-end, "
            "not that the model generalises reliably.",
            icon="⚠",
        )

        metrics_display(_active_metrics)
        st.divider()

        # Confusion matrix + ROC
        _cm_file  = "confusion_matrix_deep.png" if _show_deep else "confusion_matrix.png"
        _roc_file = "roc_curve.png"   # RF only; deep doesn't generate a separate ROC image
        col_cm, col_roc = st.columns(2)
        with col_cm:
            _cm_path = PLOTS_DIR / _cm_file
            if _cm_path.exists():
                st.image(str(_cm_path),
                         caption=f"Confusion Matrix – {_model_label}", width=400)
            else:
                st.info(f"Confusion matrix not available for {_model_label}.")
        with col_roc:
            if not _show_deep:
                roc_path = roc_curve_image(PLOTS_DIR)
                if roc_path:
                    st.image(roc_path, caption="ROC Curve – Test Set", width=400)
                    auc_val = _active_metrics.get("roc_auc")
                    if auc_val is not None:
                        st.caption(
                            f"ROC-AUC = {auc_val}.  "
                            "0.5 = random guessing; 1.0 = perfect ranking.  "
                            f"With only {n_test} test videos this curve is not statistically meaningful."
                        )
                else:
                    st.info("ROC curve not available.")
            else:
                auc_val = _active_metrics.get("roc_auc")
                st.metric("ROC-AUC", auc_val if auc_val is not None else "N/A")
                st.caption("No ROC plot generated for CNN-LSTM evaluation.")

        # Per-video test breakdown
        with st.expander(f"Per-video predictions — {_model_label}"):
            fnames = _active_metrics.get("test_filenames", [])
            trues  = _active_metrics.get("test_true_labels", [])
            preds  = _active_metrics.get("test_predictions", [])
            if fnames:
                label_map = {0: "normal", 1: "abnormal"}
                rows = [
                    {
                        "File":       f,
                        "True label": label_map.get(t, str(t)),
                        "Predicted":  label_map.get(p, str(p)),
                        "Correct":    "✓" if t == p else "✗",
                    }
                    for f, t, p in zip(fnames, trues, preds)
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info(
            "No evaluation metrics found for this model.  "
            "Run the relevant evaluation script first."
        )

    st.divider()
    dataset_summary_table(MANIFEST_PATH)

    # ── Model comparison (reads from saved file, no hardcoded scores) ─────
    comp_path = REPORTS_DIR / "model_comparison.json"
    if comp_path.exists():
        import pandas as pd
        with open(comp_path, encoding="utf-8") as _f:
            _comp = json.load(_f)

        st.subheader("Multi-Model Comparison")
        _n_train = _comp.get("train_size", "?")
        _n_val   = _comp.get("val_size", "?")
        _n_test  = _comp.get("test_size", "?")
        st.caption(
            f"All models trained on the same {_n_train} training videos "
            f"with {_n_val} validation videos.  "
            f"Final evaluation on the same {_n_test} held-out test videos.  "
            "Metrics are directly comparable because no model saw the test set during training or selection."
        )

        _model_rows = []
        _label_map = {
            "dummy_stratified":    "DummyClassifier (stratified)",
            "dummy_most_frequent": "DummyClassifier (most-frequent)",
            "logistic":            "Logistic Regression",
            "random_forest":       "Random Forest",
        }
        for _mname, _mvals in _comp["models"].items():
            _roc = _mvals.get("test_roc_auc")
            _model_rows.append({
                "Model":              _label_map.get(_mname, _mname),
                "Val Bal-Acc":        _mvals.get("val_balanced_accuracy", "N/A"),
                "Test Bal-Acc":       _mvals.get("test_balanced_accuracy", "N/A"),
                "Test ROC-AUC":       _roc if _roc is not None else "N/A",
                "TP": _mvals.get("tp","?"), "TN": _mvals.get("tn","?"),
                "FP": _mvals.get("fp","?"), "FN": _mvals.get("fn","?"),
            })
        st.dataframe(pd.DataFrame(_model_rows), hide_index=True, use_container_width=True)

        _rf = _comp["models"].get("random_forest", {})
        _dummy = _comp["models"].get("dummy_stratified", {})
        _rf_acc    = _rf.get("test_balanced_accuracy", "?")
        _dummy_acc = _dummy.get("test_balanced_accuracy", "?")
        st.warning(
            f"⚠ The trained Random Forest (test balanced-accuracy = **{_rf_acc}**) "
            f"does not outperform the DummyClassifier (**{_dummy_acc}**) on this "
            f"{_n_test}-video test set.  "
            "This indicates the model has not learned patterns that generalise to unseen videos.  "
            "The root cause is the very small labelled dataset (20 videos total).  "
            "These results are reported honestly — the pipeline is working correctly, "
            "but meaningful performance claims require substantially more labelled data.",
            icon="⚠",
        )
    else:
        st.info(
            "Model comparison report not found.  "
            "Run `python scripts/compare_models.py` to generate it."
        )


# ════════════════════════════════════════════════════════════════════════════
# Page: About
# ════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("ℹ About This Project")
    st.markdown("""
## Real-Time AI-Based Crowd Anomaly and Panic-Like Behaviour Detection

### Problem Statement
Large crowd gatherings can transition from orderly to dangerous rapidly.
Early detection of unusual motion patterns can help human operators take
preventive action. This prototype demonstrates the technical pipeline.

### Objectives
- Analyse crowd video footage for abnormal motion patterns.
- Classify videos as **normal** or **abnormal** using motion features.
- Provide a monitoring interface with configurable, human-reviewed alerts.

### Technical Pipeline (beginner-friendly explanation)

```
1. VIDEO INPUT
   ↓ Upload a video file (AVI, MOV, MP4, etc.)

2. FRAME DECODING
   ↓ OpenCV reads individual frames from the video.
   ↓ Every 10th frame is selected (configurable).
   ↓ Each frame is resized to 224×224 pixels.

3. MOTION FEATURE EXTRACTION
   ↓ Dense Optical Flow (Farneback) measures how pixels move
     between consecutive frames. High speed + chaos = suspicious.
   ↓ Frame difference measures raw pixel-level change.
   ↓ 8 statistics per frame pair, aggregated → 33-number vector.

4. MODEL PREDICTION
   ↓ StandardScaler normalises the 33 values (fitted on training data).
   ↓ Random Forest outputs P(normal) and P(abnormal) in [0, 1].

5. THRESHOLD DECISION
   ↓ Score ≥ threshold → suspicious window.
   ↓ After N consecutive suspicious windows + cooldown → ALERT.

6. HUMAN-REVIEWED ALERT
   A human operator reviews the alert and decides whether action is needed.
```

### Dataset
| Category | Total | Decodable | Used for training |
|----------|-------|-----------|-------------------|
| Normal Crowds | 15 | 12 | Yes (12 in labelled decodable set) |
| Abnormal Crowds | 8 | 8 | Yes |
| Extra (unlabelled) | 29 | 23 | No (labels not verified) |

3 non-decodable MOV files were recovered via FFmpeg conversion to MP4.

### ⚠ Limitations
- **Very small dataset** (20 labelled decodable videos: 12 normal, 8 abnormal).
  Test set has only 6 videos — metrics are not statistically reliable.
  The model performs near or below chance level on held-out data.
- The model detects **unusual motion patterns**, not confirmed panic,
  injuries, intent, fear, or emergency situations.
- 9 MOV files are unreadable by OpenCV on Windows (zlib-compressed container).
  3 labelled normal videos were recovered via FFmpeg conversion.
- No face recognition, biometric profiling, or emotion detection.

### Ethical Considerations
- All model outputs must be reviewed by a trained human operator.
- The system must not be the sole basis for any enforcement or emergency action.
- Privacy of individuals in analysed footage must be protected.
- No demographic profiling or identity tracking.

### Future Enhancements
- Larger annotated dataset (UCF-Crime, ShanghaiTech Crowd).
- Temporal deep learning (LSTM / 3D-CNN) on frame sequences.
- Crowd density estimation and heatmap overlays.
- Multi-camera fusion.
- Uncertainty quantification.
""")
    st.divider()
    st.caption(
        "Built with Python · OpenCV · scikit-learn · Streamlit  |  "
        "Academic mini-project — educational purposes only."
    )
