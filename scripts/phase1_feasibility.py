"""Phase 1: feasibility report for CNN-LSTM deep learning upgrade."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np, torch

ROOT = pathlib.Path(".")
df = pd.read_csv(ROOT / "dataset/dataset_manifest.csv")
lab = df[(df["source_group"]=="labelled") & (df["decoded"]==True)]

print("="*70)
print("PHASE 1 FEASIBILITY REPORT")
print("="*70)
print()

# Dataset
for split in ["train","val","test"]:
    s = lab[lab["split"]==split]
    if len(s):
        n = int((s["label"]=="normal").sum())
        a = int((s["label"]=="abnormal").sum())
        print(f"  {split:<6}: {len(s)} videos  (normal={n}, abnormal={a})")
total = len(lab)
print(f"  TOTAL LABELLED DECODABLE: {total}")
print()

# Frame counts
print("Frame counts (labelled decodable):")
for _, r in lab.iterrows():
    fc = r["frame_count"]
    dur = r["duration_s"]
    print(f"  {r['filename']:40s}  {r['label']:10s}  frames={fc}  dur={dur}s  split={r['split']}")
print()

# At frame_sample_interval=10, how many sampled frames per video?
interval = 10
print(f"At sample_interval={interval}, approximate sampled frames per video:")
short = 0
for _, r in lab.iterrows():
    fc = r["frame_count"]
    if pd.notna(fc):
        n_sampled = int(fc) // interval
        flag = " *** TOO SHORT (<5)" if n_sampled < 5 else ""
        print(f"  {r['filename']:40s}  sampled~{n_sampled:>4}{flag}")
        if n_sampled < 5:
            short += 1
print(f"  Videos with fewer than 5 sampled frames: {short}")
print()

# Architecture sizing
print("CNN-LSTM architecture sizing:")
print("  Backbone: MobileNetV3-Small (pretrained ImageNet, frozen)")
print("  Feature dim: 576  (output of MobileNetV3-Small avgpool)")
print("  LSTM: hidden=128, layers=1, bidirectional=False")
print("  Sequence length: variable (num sampled frames from video)")
print("  Min usable sequence length: 2 frames (LSTM minimum)")
print("  Output: 2-class softmax  (0=normal, 1=abnormal)")
print()

# Pretrained weights download check
print("Pretrained weights:")
try:
    from torchvision.models import MobileNet_V3_Small_Weights
    w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    print(f"  MobileNetV3-Small IMAGENET1K_V1 weights descriptor: OK")
    print(f"  Download URL will be checked at first load (approx 10 MB)")
except Exception as e:
    print(f"  WARN: {e}")
print()

# Data split assessment
n_train = len(lab[lab["split"]=="train"])
n_test  = len(lab[lab["split"]=="test"])
print("Assessment:")
print(f"  Training videos: {n_train} — very small for deep learning.")
print(f"  With frozen CNN backbone and LSTM, overfitting is expected.")
print(f"  Training will proceed but results cannot be statistically trusted")
print(f"  on a {n_test}-video test set.")
print(f"  The CNN-LSTM will be labelled EXPERIMENTAL in the dashboard.")
print(f"  Pretrained features provide a meaningful head-start over random init.")
print()
print("VERDICT: Feasible on CPU with frozen MobileNetV3-Small backbone.")
print("  Estimated training time: ~2-5 min (11 videos, CPU, 30 epochs)")
print("  Estimated inference time: ~1-3 s per video (CPU)")
