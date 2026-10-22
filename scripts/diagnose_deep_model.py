"""
Diagnose why the CNN-LSTM predicts all-normal on the test set.
Runs inference on EVERY labelled video (train + val + test) and prints scores.
"""
import sys, torch, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.models.deep_model import load_deep_model, LABEL_MAP, LABEL_NAMES
from src.models.deep_dataset import _extract_frame_features
from src.data.inventory import load_inventory
from src.utils.paths import load_config, get_path, PROJECT_ROOT

cfg = load_config()
ckpt = get_path(cfg, "training", "model_output_dir") / "deep_model.pth"
model = load_deep_model(ckpt)
model.eval()

df = load_inventory(PROJECT_ROOT / cfg["dataset"]["manifest_path"])
labelled = df[(df["source_group"] == "labelled") & (df["decoded"] == True)]

print("=== ALL LABELLED VIDEOS — CNN-LSTM scores from checkpoint (epoch 9) ===")
print(f"{'File':42s} {'Split':6s} {'True':10s} {'Pred':10s} {'P(abn)':8s} {'Correct'}")
print("-" * 90)

n_correct = 0
n_total = 0
for _, row in labelled.iterrows():
    p = pathlib.Path(row["path"])
    feat = _extract_frame_features(p, model, 10, 150, torch.device("cpu"))
    if feat is None:
        print(f"  {row['filename']:42s} {str(row['split']):6s} {row['label']:10s} FEAT_FAIL")
        continue
    x = feat.unsqueeze(0)
    lengths = torch.tensor([feat.shape[0]])
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = LABEL_NAMES[int(logits.argmax(dim=1).item())]
    correct = pred == row["label"]
    n_correct += int(correct)
    n_total += 1
    print(f"  {row['filename']:42s} {str(row['split']):6s} {row['label']:10s} {pred:10s} "
          f"{probs[1].item():8.3f}  {'OK' if correct else 'WRONG'}")

print("-" * 90)
print(f"Overall accuracy on all labelled: {n_correct}/{n_total}")

# Show the classifier head weights and bias
import torch
ckpt_data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
sd = ckpt_data["model_state_dict"]
print()
print("Classifier head:")
print("  weight:", sd["classifier.weight"].tolist())
print("  bias:  ", sd["classifier.bias"].tolist())
print()
print("DIAGNOSIS:")
print("  If all P(abn) < 0.5, the model learned to predict normal for everything.")
print("  This is a collapse caused by class imbalance + tiny dataset.")
print("  Fix: lower LR, stronger class weights, or per-class balanced sampling.")
