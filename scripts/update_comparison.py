"""Add CNN-LSTM results to the model comparison report."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(".")
comp_path = ROOT / "outputs/reports/model_comparison.json"
deep_path = ROOT / "outputs/reports/evaluation_metrics_deep.json"

comp = json.loads(comp_path.read_text())
deep = json.loads(deep_path.read_text())

comp["models"]["cnn_lstm"] = {
    "val_balanced_accuracy": "N/A",
    "test_balanced_accuracy": deep["balanced_accuracy"],
    "test_roc_auc": deep.get("roc_auc"),
    "tp": deep["tp"], "tn": deep["tn"],
    "fp": deep["fp"], "fn": deep["fn"],
    "test_predictions": deep.get("test_predictions", []),
    "note": "Frozen MobileNetV3-Small + LSTM, 9 epochs (val_loss=0.3563), EXPERIMENTAL",
}

with open(comp_path, "w") as f:
    json.dump(comp, f, indent=2)

print("model_comparison.json updated.")
print()
for name, vals in comp["models"].items():
    ba = vals["test_balanced_accuracy"]
    roc = vals["test_roc_auc"]
    print(f"  {name:30s}: test_balacc={ba}  roc={roc}")
