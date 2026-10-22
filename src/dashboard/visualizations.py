"""
src/dashboard/visualizations.py
--------------------------------
Chart and plot helper functions for the Streamlit dashboard.
All charts use real data; none produce hardcoded / simulated values.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


def label_distribution_chart(manifest_path: Path) -> go.Figure:
    """Pie chart of label distribution from the manifest."""
    if not manifest_path.exists():
        return go.Figure().update_layout(title="Manifest not found")

    df = pd.read_csv(manifest_path)
    counts = df.groupby("label").size().reset_index(name="count")

    fig = px.pie(
        counts,
        names="label",
        values="count",
        title="Dataset Label Distribution",
        color_discrete_map={
            "normal":   "#2ecc71",
            "abnormal": "#e74c3c",
            "unknown":  "#95a5a6",
        },
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def decodability_chart(manifest_path: Path) -> go.Figure:
    """Bar chart showing decodable vs non-decodable per label."""
    if not manifest_path.exists():
        return go.Figure()

    df = pd.read_csv(manifest_path)
    grp = (
        df.groupby(["label", "decoded"])
        .size()
        .reset_index(name="count")
    )
    grp["status"] = grp["decoded"].map({True: "Decodable", False: "Non-decodable"})

    fig = px.bar(
        grp,
        x="label",
        y="count",
        color="status",
        title="Decodable vs Non-decodable Videos per Label",
        color_discrete_map={"Decodable": "#2ecc71", "Non-decodable": "#e74c3c"},
        barmode="stack",
    )
    return fig


def score_history_chart(scores: list[float], threshold: float) -> go.Figure:
    """Line chart of anomaly scores in a live monitoring session."""
    fig = go.Figure()

    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color="orange", annotation_text=f"Alert threshold ({threshold:.2f})",
    )
    fig.add_trace(go.Scatter(
        y=scores,
        mode="lines",
        line=dict(color="steelblue", width=2),
        fill="tozeroy",
        fillcolor="rgba(70,130,180,0.15)",
        name="Score",
    ))
    fig.update_layout(
        title="Real-Time Anomaly Score",
        xaxis_title="Frame window",
        yaxis_title="P(abnormal)",
        yaxis=dict(range=[0, 1]),
        height=260,
        margin=dict(l=30, r=20, t=40, b=30),
    )
    return fig


def roc_curve_image(plots_dir: Path):
    """Return the ROC curve image path if it exists, else None."""
    p = plots_dir / "roc_curve.png"
    return str(p) if p.exists() else None
