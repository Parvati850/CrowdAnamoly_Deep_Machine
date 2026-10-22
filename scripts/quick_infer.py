"""
Quick inference test on videos from the actual test split.
Reads the current manifest to find which videos are in the test set,
then runs single-video inference on each one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, PROJECT_ROOT
from src.utils.logging_config import setup_logging
from src.models.predict import predict_video
import pandas as pd

cfg = load_config()
setup_logging("WARNING")

# Read test split from the manifest
manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
df = pd.read_csv(manifest_path)
test_df = df[df["split"] == "test"]

print(f"\nTest split: {len(test_df)} videos")
print("-" * 80)
print(f"{'Video':<40} {'True':<10} {'Predicted':<12} {'Score':>8}  {'OK?'}")
print("-" * 80)

correct = 0
for _, row in test_df.iterrows():
    path = Path(row["path"])
    true_label = row["label"]
    r = predict_video(path, cfg=cfg)
    if r["error"]:
        print(f"{row['filename']:<40} {true_label:<10} ERROR: {r['error']}")
        continue
    match = "CORRECT" if r["label"] == true_label else "WRONG"
    if r["label"] == true_label:
        correct += 1
    print(f"{row['filename']:<40} {true_label:<10} {r['label']:<12} {r['score_abnormal']:>8.4f}  {match}")

print("-" * 80)
print(f"Correct: {correct}/{len(test_df)}")
print()
