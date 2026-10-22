"""Audit verification script — read-only checks."""
import sys, pathlib, numpy as np, pandas as pd, json, pickle, inspect, re

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
ROOT = pathlib.Path(__file__).resolve().parent.parent

results = {}

# ── 1. Dataset manifest ───────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "dataset/dataset_manifest.csv")
results["total_videos"]     = len(df)
results["normal"]           = int((df["label"]=="normal").sum())
results["abnormal"]         = int((df["label"]=="abnormal").sum())
results["unknown"]          = int((df["label"]=="unknown").sum())
results["decodable"]        = int(df["decoded"].sum())
results["non_decodable"]    = int((~df["decoded"]).sum())
results["non_dec_files"]    = df[~df["decoded"]]["filename"].tolist()

# Split distribution
labelled = df[df["source_group"]=="labelled"]
split_dist = {}
for s in ["train","val","test"]:
    sub = labelled[labelled["split"]==s]
    split_dist[s] = {"total": len(sub),
                     "normal": int((sub["label"]=="normal").sum()),
                     "abnormal": int((sub["label"]=="abnormal").sum())}
results["split_distribution"] = split_dist

# Leakage check
split_df = df[df["split"].notna()]
leakage = split_df.groupby("path")["split"].nunique()
results["leakage_videos"] = int((leakage > 1).sum())

# ── 2. Feature caches ─────────────────────────────────────────────────────────
feat_info = {}
for s in ["train","val","test"]:
    p = ROOT / f"generated_data/features/features_{s}.npz"
    if p.exists():
        d = np.load(str(p), allow_pickle=True)
        X = d["X"]
        y = d["y"]
        classes, counts = np.unique(y, return_counts=True)
        feat_info[s] = {"shape": list(X.shape),
                        "feature_count": X.shape[1],
                        "class_dist": {int(c): int(n) for c,n in zip(classes,counts)}}
    else:
        feat_info[s] = "MISSING"
results["feature_caches"] = feat_info

# ── 3. Model checkpoint ───────────────────────────────────────────────────────
ckpt = ROOT / "models/baseline_model.pkl"
results["model_checkpoint_exists"] = ckpt.exists()
if ckpt.exists():
    with open(ckpt,"rb") as f:
        pipeline = pickle.load(f)
    results["model_type"] = type(pipeline).__name__
    results["model_steps"] = [s for s,_ in pipeline.steps]
    # Verify predict_proba exists
    results["has_predict_proba"] = hasattr(pipeline, "predict_proba")

# ── 4. Evaluation metrics ─────────────────────────────────────────────────────
mpath = ROOT / "outputs/reports/evaluation_metrics.json"
if mpath.exists():
    m = json.loads(mpath.read_text())
    results["eval_balanced_accuracy"] = m.get("balanced_accuracy")
    results["eval_roc_auc"]           = m.get("roc_auc")
    results["eval_test_size"]         = len(m.get("test_filenames", []))
    results["eval_tp"] = m.get("tp"); results["eval_fp"] = m.get("fp")
    results["eval_fn"] = m.get("fn"); results["eval_tn"] = m.get("tn")
    # Verify confusion matrix matches predictions
    trues = m.get("test_true_labels", [])
    preds = m.get("test_predictions", [])
    manual_tp = sum(1 for t,p in zip(trues,preds) if t==1 and p==1)
    manual_tn = sum(1 for t,p in zip(trues,preds) if t==0 and p==0)
    manual_fp = sum(1 for t,p in zip(trues,preds) if t==0 and p==1)
    manual_fn = sum(1 for t,p in zip(trues,preds) if t==1 and p==0)
    results["confusion_matrix_verified"] = (
        manual_tp == m.get("tp") and manual_tn == m.get("tn") and
        manual_fp == m.get("fp") and manual_fn == m.get("fn")
    )
else:
    results["eval_metrics"] = "MISSING"

# ── 5. Hardcoded paths in source ──────────────────────────────────────────────
hardcoded = []
for pyfile in ROOT.rglob("*.py"):
    if ".venv" in str(pyfile) or "__pycache__" in str(pyfile):
        continue
    text = pyfile.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(r'D:[/\\]7th', line) and "config" not in str(pyfile):
            hardcoded.append(f"{pyfile.name}:{i}: {line.strip()[:80]}")
results["hardcoded_paths_outside_config"] = hardcoded

# ── 6. Placeholder / TODO scan ────────────────────────────────────────────────
todos = []
for pyfile in ROOT.rglob("*.py"):
    if ".venv" in str(pyfile) or "__pycache__" in str(pyfile):
        continue
    for i, line in enumerate(pyfile.read_text(errors="ignore").splitlines(), 1):
        if re.search(r'\bTODO\b|\bFIXME\b|\bNotImplementedError\b|raise NotImplemented\b', line):
            todos.append(f"{pyfile.name}:{i}: {line.strip()[:80]}")
results["todos_placeholders"] = todos

# ── 7. Feature vector consistency ─────────────────────────────────────────────
from src.features.motion_features import feature_names
fn_count = len(feature_names())
results["feature_names_count"] = fn_count
results["feature_vector_consistent"] = (
    feat_info.get("train", {}).get("feature_count") == fn_count
    if isinstance(feat_info.get("train"), dict) else False
)

# ── Print report ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  AUDIT CHECKS REPORT")
print("="*60)
for k, v in results.items():
    if isinstance(v, list) and len(v) > 10:
        print(f"  {k}: [{len(v)} items]")
    elif isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")
    else:
        print(f"  {k}: {v}")
print("="*60)
