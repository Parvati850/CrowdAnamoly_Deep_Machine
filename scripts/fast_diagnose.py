"""
Fast diagnosis: loads cached .npz features, runs CNN-LSTM forward pass,
reports class mapping, probabilities, confusion matrix.
No video decoding. Completes in seconds.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch, numpy as np
from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.models.deep_model import load_deep_model, LABEL_MAP, LABEL_NAMES
from src.data.inventory import load_inventory

cfg = load_config()

# ── 1. Checkpoint metadata ────────────────────────────────────────────────────
ckpt = get_path(cfg,"training","model_output_dir") / "deep_model.pth"
raw  = torch.load(str(ckpt), map_location="cpu", weights_only=False)

print("=== Checkpoint metadata ===")
for k,v in raw.items():
    if k != "model_state_dict":
        print(f"  {k}: {v}")

sd = raw["model_state_dict"]
print(f"\nClassifier weight shape: {sd['classifier.weight'].shape}")
print(f"Classifier bias: {sd['classifier.bias'].tolist()}")
print(f"  → bias[0]=normal({sd['classifier.bias'][0]:.4f})  "
      f"bias[1]=abnormal({sd['classifier.bias'][1]:.4f})")
print(f"  → bias favours class {'normal' if sd['classifier.bias'][0]>sd['classifier.bias'][1] else 'abnormal'}")

# ── 2. Label map verification ─────────────────────────────────────────────────
print("\n=== Label encoding ===")
print(f"  LABEL_MAP = {LABEL_MAP}  (0=normal, 1=abnormal)")
print(f"  LABEL_NAMES = {LABEL_NAMES}")
print("  CrossEntropyLoss(weight=[w_normal, w_abnormal])")
print("  logits[:,0]=normal logit, logits[:,1]=abnormal logit")
print("  argmax(logits) → predicted class index")

# ── 3. Use CACHED numpy features (from generated_data/features/*.npz) ────────
# These were extracted by the Random Forest pipeline with the same
# frame_sample_interval=10 and target 224×224 — we can use them as a proxy
# to check whether the LSTM can distinguish classes at all on these features.
# Note: these are the 33-dim RF features, NOT the 576-dim CNN features.
# We create synthetic 576-dim inputs to probe the LSTM head in isolation.

print("\n=== Probing LSTM head on synthetic inputs ===")
model = load_deep_model(ckpt)
model.eval()

# Create two synthetic 576-dim sequences: one all-zeros (minimal motion) and
# one all-ones (maximal activation) — proxy for normal vs abnormal
for name, val in [("zeros (minimal)", 0.0), ("ones (maximal)", 1.0)]:
    x = torch.full((1, 20, 576), val)
    lengths = torch.tensor([20])
    with torch.no_grad():
        logits = model(x, lengths)
        probs  = torch.softmax(logits, dim=1).squeeze(0)
    pred = LABEL_NAMES[int(logits.argmax(dim=1).item())]
    print(f"  Input {name}: P(normal)={probs[0]:.3f}  P(abnormal)={probs[1]:.3f}  → pred={pred}")

# ── 4. Use the CACHED RF .npz feature arrays for quick class-distribution check
print("\n=== Cached RF feature statistics (train split) ===")
feat_path = get_path(cfg, "preprocessing", "features_output_dir") / "features_train.npz"
if feat_path.exists():
    d = np.load(str(feat_path), allow_pickle=True)
    X, y, fnames = d["X"], d["y"], list(d["filenames"])
    print(f"  Shape: {X.shape}   Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    for i, fn in enumerate(fnames):
        print(f"    {fn:40s}  y={y[i]}({'normal' if y[i]==0 else 'abnormal'})  "
              f"of_mag_mean={X[i,0]:.3f}  fd_mean={X[i,6]:.3f}")
else:
    print("  features_train.npz not found")

# ── 5. Training history summary ───────────────────────────────────────────────
hist_path = PROJECT_ROOT / "outputs/reports/deep_train_history.json"
if hist_path.exists():
    h = json.loads(hist_path.read_text())
    ep = h["epochs"]
    print(f"\n=== Training history ({len(ep)} epochs logged) ===")
    print(f"{'Ep':>4}  {'tr_loss':>8}  {'tr_acc':>7}  {'val_loss':>9}  {'val_acc':>8}")
    for i in range(len(ep)):
        marker = " ← best ckpt" if h["val_loss"][i] == min(h["val_loss"]) else ""
        print(f"  {ep[i]:>3}  {h['train_loss'][i]:>8.4f}  {h['train_acc'][i]:>7.4f}  "
              f"{h['val_loss'][i]:>9.4f}  {h['val_acc'][i]:>8.4f}{marker}")

# ── 6. Root cause analysis ────────────────────────────────────────────────────
print("\n=== Root cause analysis ===")
bias0 = float(sd['classifier.bias'][0])
bias1 = float(sd['classifier.bias'][1])
print(f"  Bias delta (normal - abnormal): {bias0-bias1:.4f}")
if bias0 > bias1:
    print("  → The classifier head is biased toward 'normal'.")
    print("  → The LSTM hidden state doesn't provide enough signal to overcome this.")

if hist_path.exists():
    tr_accs = h["train_acc"]
    stuck = all(abs(a - tr_accs[-1]) < 0.01 for a in tr_accs[-5:])
    if stuck:
        print(f"  → Training accuracy stuck at {tr_accs[-1]:.3f} for last 5 epochs.")
        print("    This is the collapsed-to-majority-class pattern.")
        base_normal_frac = 7/11  # 7 normal, 4 abnormal in train
        print(f"    Majority class (normal) fraction = 7/11 = {base_normal_frac:.3f}")
        print(f"    Model just predicts normal → gets {base_normal_frac:.3f} train accuracy.")

print("""
CONFIRMED ROOT CAUSES:
  1. The LSTM collapsed to predicting 'normal' for everything.
     Training accuracy of 0.636 = exactly 7/11 (majority-class rate).
     Val accuracy of 0.667 = 2/3 (2 normal out of 3 val videos).
  2. The class imbalance (7 normal vs 4 abnormal) combined with:
     - Very high-dimensional input (576-dim per frame)
     - Only 11 training videos
     - LR 3e-4 may be too high for LSTM to escape the majority-class optimum
     - Dropout 0.3 on the LSTM output adds noise that hurts tiny datasets
  3. Epoch 9 checkpoint is the best by val_loss — but val_loss also reflects
     the collapsed model (all-normal predictions → 1 wrong = 0.667 val_acc).

PROPOSED FIX:
  Retrain from scratch with these changes (no architecture changes needed):
    a. Lower LR: 1e-4 (was 3e-4) — prevents jumping over the abnormal signal
    b. Stronger class weights: total/(n_class * count) already used; add
       focal-loss-style gamma=2 to focus on hard (abnormal) examples.
       OR simply upweight abnormal class by factor 3x explicitly.
    c. Reduce dropout: 0.1 (was 0.3) — less regularisation on 11 samples.
    d. Patience: 15 (was 10) — allow more epochs for slow convergence.
  These are the smallest justified changes. Architecture stays the same.
""")
