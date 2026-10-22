"""Quick verification of evaluation/comparison data files."""
import json, pathlib

ROOT = pathlib.Path(".")
em = json.loads((ROOT / "outputs/reports/evaluation_metrics.json").read_text())
print("evaluation_metrics.json keys:", list(em.keys()))
print("  balanced_accuracy:", em["balanced_accuracy"])
print("  roc_auc:", em["roc_auc"])
print("  test_filenames:", em["test_filenames"])
print()

mc = json.loads((ROOT / "outputs/reports/model_comparison.json").read_text())
print("model_comparison.json keys:", list(mc.keys()))
print("  models:", list(mc["models"].keys()))
for name, vals in mc["models"].items():
    print(f"  {name}: test_balacc={vals['test_balanced_accuracy']} roc={vals['test_roc_auc']}")
print()

tm = json.loads((ROOT / "outputs/reports/train_metadata.json").read_text())
print("train_metadata.json:", {k: tm[k] for k in ["classifier_type","n_train","n_val","train_accuracy_on_trainset"]})
