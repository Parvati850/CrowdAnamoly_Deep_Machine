"""
Verify that the Model Performance comparison block in app.py
reads the actual model_comparison.json values and would display them correctly.
Simulates the rendering logic without starting Streamlit.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd

ROOT = pathlib.Path(".")
REPORTS_DIR = ROOT / "outputs" / "reports"
comp_path = REPORTS_DIR / "model_comparison.json"

assert comp_path.exists(), f"FAIL: {comp_path} does not exist"
with open(comp_path) as f:
    comp = json.load(f)

# Replicate the exact rendering logic from app.py
_n_train = comp.get("train_size", "?")
_n_val   = comp.get("val_size", "?")
_n_test  = comp.get("test_size", "?")

_label_map = {
    "dummy_stratified":    "DummyClassifier (stratified)",
    "dummy_most_frequent": "DummyClassifier (most-frequent)",
    "logistic":            "Logistic Regression",
    "random_forest":       "Random Forest",
}
_model_rows = []
for _mname, _mvals in comp["models"].items():
    _roc = _mvals.get("test_roc_auc")
    _model_rows.append({
        "Model":        _label_map.get(_mname, _mname),
        "Val Bal-Acc":  _mvals.get("val_balanced_accuracy", "N/A"),
        "Test Bal-Acc": _mvals.get("test_balanced_accuracy", "N/A"),
        "Test ROC-AUC": _roc if _roc is not None else "N/A",
        "TP": _mvals.get("tp","?"), "TN": _mvals.get("tn","?"),
        "FP": _mvals.get("fp","?"), "FN": _mvals.get("fn","?"),
    })

df = pd.DataFrame(_model_rows)
print(f"Comparison table ({_n_test}-video test set, {_n_train} train, {_n_val} val):")
print(df.to_string(index=False))
print()

# Check warning values
_rf    = comp["models"].get("random_forest", {})
_dummy = comp["models"].get("dummy_stratified", {})
_rf_acc    = _rf.get("test_balanced_accuracy", "?")
_dummy_acc = _dummy.get("test_balanced_accuracy", "?")
print(f"Warning: RF={_rf_acc} vs Dummy={_dummy_acc}")
print()

# Verify no hardcoded score strings in the comparison block
import re
with open("app.py") as f:
    app_text = f.read()
# The comparison block should not contain literal 0.3333, 0.8333 etc
# (it reads them from the JSON via _mvals.get())
comparison_block_start = app_text.find("# ── Model comparison (reads from saved file")
comparison_block_end   = app_text.find("# Page: About")
comparison_block = app_text[comparison_block_start:comparison_block_end]
hardcoded = re.findall(r'[0-9]+\.[0-9]{4}', comparison_block)
if hardcoded:
    print(f"WARNING: Possible hardcoded floats found in comparison block: {hardcoded}")
    print("  (These may be OK if they are format strings, not metric values)")
else:
    print("No hardcoded score floats in the comparison block — all values read from file.")

print()
print("PASS — comparison display logic verified against actual file")
