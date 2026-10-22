"""
src/dashboard/components.py
----------------------------
Reusable Streamlit UI components for the CrowdAnomalyDetection dashboard.
All functions take real data from the inference pipeline; nothing is hardcoded.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


def model_status_banner(model_loaded: bool, checkpoint_path: str = "") -> None:
    """Show a prominent banner about whether the model is trained and loaded."""
    if model_loaded:
        st.success(f"✅ Model loaded from: `{checkpoint_path}`")
    else:
        st.error(
            "❌ No trained model found.  "
            "Please run `python scripts/train_model.py` first, "
            "then refresh this page."
        )


def score_gauge(score: float, threshold: float) -> None:
    """Display the anomaly score as a colour-coded metric with context."""
    colour = "normal" if score < threshold else "inverse"
    label  = "🟢 NORMAL"   if score < threshold else "🔴 ANOMALY DETECTED"
    st.metric(
        label="Anomaly Score (P(abnormal))",
        value=f"{score:.3f}",
        delta=f"Threshold: {threshold:.2f}",
        delta_color=colour,
    )
    st.markdown(f"**Status: {label}**")
    st.progress(min(float(score), 1.0))
    st.caption(
        "⚠ This is a prototype decision-support tool. "
        "The score reflects model confidence based on motion features only. "
        "Do NOT treat this as a confirmed emergency detection system."
    )


def window_score_chart(windows: list[dict]) -> None:
    """Plot a line chart of anomaly scores across video windows."""
    if not windows:
        st.info("No analysis windows to display.")
        return

    import plotly.graph_objects as go

    times  = [w.get("start_time_s", i) for i, w in enumerate(windows)]
    scores = [w["score"] for w in windows]
    labels = [w["label"] for w in windows]

    colours = ["red" if l == "abnormal" else "green" for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=scores, mode="lines+markers",
        marker=dict(color=colours, size=8),
        line=dict(color="steelblue", width=2),
        name="Anomaly Score",
    ))
    fig.update_layout(
        title="Anomaly Score Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Score (0=normal, 1=abnormal)",
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, width="stretch")


def event_log_table(events: list) -> None:
    """Display the alert event log as a styled table."""
    if not events:
        st.info("No alerts fired yet.")
        return

    rows = []
    for i, e in enumerate(events):
        rows.append({
            "#":           i + 1,
            "Time":        e.timestamp,
            "Source":      e.source,
            "Score":       f"{e.score_abnormal:.3f}",
            "Threshold":   f"{e.threshold:.2f}",
            "Label":       e.label,
            "Acknowledged": "✓" if e.acknowledged else "",
            "Dismissed":   "✓" if e.dismissed else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def metrics_display(metrics: dict) -> None:
    """Show evaluation metrics in a structured layout."""
    if "error" in metrics:
        st.warning(metrics["error"])
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.3f}")
    col2.metric("ROC-AUC", f"{metrics.get('roc_auc') or 'N/A'}")
    col3.metric("False Positives", metrics.get("fp", "?"))
    col4.metric("False Negatives", metrics.get("fn", "?"))

    st.markdown("**Per-class results:**")
    per_class = metrics.get("per_class", {})
    rows = []
    for label in ["normal", "abnormal"]:
        r = per_class.get(label, {})
        rows.append({
            "Class":     label,
            "Precision": f"{r.get('precision', 0):.3f}",
            "Recall":    f"{r.get('recall', 0):.3f}",
            "F1-score":  f"{r.get('f1-score', 0):.3f}",
            "Support":   int(r.get("support", 0)),
        })
    st.table(pd.DataFrame(rows))

    st.caption(
        "⚠ The test set contains very few videos. "
        "These metrics are indicative only and will vary significantly "
        "with additional data."
    )


def confusion_matrix_image(plots_dir: Path) -> None:
    """Display a saved confusion matrix image if available."""
    img_path = plots_dir / "confusion_matrix.png"
    if img_path.exists():
        st.image(str(img_path), caption="Confusion Matrix – Test Set", width=400)
    else:
        st.info("Confusion matrix not available. Run evaluate_model.py first.")


def dataset_summary_table(manifest_path: Path) -> None:
    """Show the dataset manifest summary."""
    if not manifest_path.exists():
        st.warning("Dataset manifest not found. Run inspect_dataset.py first.")
        return

    df = pd.read_csv(manifest_path)
    st.markdown(f"**Total videos:** {len(df)}")

    summary = df.groupby(["label", "source_group"]).agg(
        count=("filename", "count"),
        decodable=("decoded", "sum"),
    ).reset_index()
    st.dataframe(summary, width="stretch", hide_index=True)

    non_dec = df[~df["decoded"]][["filename", "folder", "label", "decode_error"]]
    if not non_dec.empty:
        with st.expander(f"⚠ {len(non_dec)} non-decodable videos"):
            st.dataframe(non_dec, width="stretch", hide_index=True)
