"""
scripts/compare_models.py
--------------------------
Fair baseline comparison: DummyClassifier, LogisticRegression, RandomForest
all trained on the same features with the same group-aware split.
Uses validation set for model selection, test set ONLY for final evaluation.

Also inspects feature distributions and per-video test predictions.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
)

from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.utils.logging_config import setup_logging, get_logger
from src.features.video_features import load_features
from src.features.motion_features import feature_names

cfg = load_config()
setup_logging("WARNING")
logger = get_logger("compare_models")

# ── Load features ────────────────────────────────────────────────────────────
X_train, y_train, fn_train = load_features(cfg, split="train")
X_val, y_val, fn_val       = load_features(cfg, split="val")
X_test, y_test, fn_test    = load_features(cfg, split="test")

print(f"Train: {X_train.shape}  classes={dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Val:   {X_val.shape}  classes={dict(zip(*np.unique(y_val, return_counts=True)))}")
print(f"Test:  {X_test.shape}  classes={dict(zip(*np.unique(y_test, return_counts=True)))}")
print()

# ── Feature distribution comparison ─────────────────────────────────────────
print("=" * 80)
print("FEATURE DISTRIBUTION: Normal vs Abnormal (training set)")
print("=" * 80)
names = feature_names()
train_df = pd.DataFrame(X_train, columns=names)
train_df["label"] = ["normal" if y == 0 else "abnormal" for y in y_train]

for col in names[:8]:  # Show the 8 "mean" features (most interpretable)
    n_stats = train_df[train_df["label"] == "normal"][col]
    a_stats = train_df[train_df["label"] == "abnormal"][col]
    print(f"  {col:30s}  normal: mean={n_stats.mean():.3f} std={n_stats.std():.3f}  "
          f"abnormal: mean={a_stats.mean():.3f} std={a_stats.std():.3f}")

print()
print("=" * 80)
print("PER-VIDEO TEST FEATURES (key features)")
print("=" * 80)
test_df = pd.DataFrame(X_test, columns=names)
test_df["filename"] = fn_test
test_df["true_label"] = ["normal" if y == 0 else "abnormal" for y in y_test]
key_features = ["of_mag_mean_mean", "of_direction_entropy_mean", "fd_mean_mean", "n_frame_pairs"]
for _, row in test_df.iterrows():
    print(f"  {row['filename']:40s}  {row['true_label']:10s}  "
          f"of_mag={row['of_mag_mean_mean']:.3f}  "
          f"entropy={row['of_direction_entropy_mean']:.3f}  "
          f"fd_mean={row['fd_mean_mean']:.3f}  "
          f"pairs={row['n_frame_pairs']:.0f}")

print()

# ── Model comparison ────────────────────────────────────────────────────────
models = {
    "dummy_stratified": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DummyClassifier(strategy="stratified", random_state=42)),
    ]),
    "dummy_most_frequent": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ]),
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=42)),
    ]),
}

print("=" * 80)
print("MODEL COMPARISON (train→fit, val→select, test→final)")
print("=" * 80)
print(f"{'Model':<25} {'Val BalAcc':>10} {'Test BalAcc':>11} {'Test ROC-AUC':>12} "
      f"{'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3}")
print("-" * 80)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)

    # Validation
    val_preds = model.predict(X_val)
    val_balacc = balanced_accuracy_score(y_val, val_preds)

    # Test
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    test_balacc = balanced_accuracy_score(y_test, test_preds)

    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    roc = None
    if test_probs is not None and len(np.unique(y_test)) == 2:
        try:
            roc = round(float(roc_auc_score(y_test, test_probs[:, 1])), 4)
        except Exception:
            pass

    results[name] = {
        "val_balanced_accuracy": round(val_balacc, 4),
        "test_balanced_accuracy": round(test_balacc, 4),
        "test_roc_auc": roc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "test_predictions": test_preds.tolist(),
        "test_true": y_test.tolist(),
    }

    roc_str = f"{roc:.4f}" if roc is not None else "N/A"
    print(f"{name:<25} {val_balacc:>10.4f} {test_balacc:>11.4f} {roc_str:>12} "
          f"{tp:>3} {tn:>3} {fp:>3} {fn:>3}")

print()

# ── Per-video test predictions for each model ──────────────────────────────
print("=" * 80)
print("PER-VIDEO TEST PREDICTIONS")
print("=" * 80)
print(f"{'File':<40} {'True':<10}", end="")
for name in models:
    print(f" {name[:12]:>13}", end="")
print()
print("-" * 80)

for i, fn in enumerate(fn_test):
    true_label = "normal" if y_test[i] == 0 else "abnormal"
    print(f"{fn:<40} {true_label:<10}", end="")
    for name in models:
        pred = results[name]["test_predictions"][i]
        pred_label = "normal" if pred == 0 else "abnormal"
        correct = "✓" if pred == y_test[i] else "✗"
        print(f" {pred_label:>12}{correct}", end="")
    print()

print()

# ── Save comparison report ─────────────────────────────────────────────────
report_path = PROJECT_ROOT / "outputs" / "reports" / "model_comparison.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump({
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "feature_count": X_train.shape[1],
        "test_filenames": fn_test,
        "models": results,
    }, f, indent=2, default=str)
print(f"Report saved to: {report_path}")
