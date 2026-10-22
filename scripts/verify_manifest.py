"""Verify the updated manifest after codec conversion."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "dataset/dataset_manifest.csv")

print(f"Total: {len(df)}")
print(f"Normal: {(df['label']=='normal').sum()}")
print(f"Abnormal: {(df['label']=='abnormal').sum()}")
print(f"Unknown: {(df['label']=='unknown').sum()}")
print(f"Decodable: {df['decoded'].sum()}")
print(f"Non-decodable: {(~df['decoded']).sum()}")
print()

lab = df[(df['source_group']=='labelled') & (df['decoded']==True)]
print(f"Labelled decodable:")
print(f"  Total: {len(lab)}")
print(f"  Normal: {(lab['label']=='normal').sum()}")
print(f"  Abnormal: {(lab['label']=='abnormal').sum()}")
print()

print("Converted videos in manifest:")
conv = df[df['filename'].str.contains('_converted', na=False)]
print(f"  Count: {len(conv)}")
for _, r in conv.iterrows():
    print(f"  {r['filename']:40s} label={r['label']:10s} decoded={r['decoded']}  dur={r['duration_s']}s")
print()

print(f"Duplicate filenames: {df.duplicated('filename').sum()}")
print(f"Duplicate paths: {df.duplicated('path').sum()}")
