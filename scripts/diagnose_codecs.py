"""Diagnose the 9 non-decodable videos — read-only inspection."""
import sys, pathlib, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "dataset/dataset_manifest.csv")
bad = df[df["decoded"] == False]

print(f"Non-decodable files: {len(bad)}")
print("=" * 80)
for _, r in bad.iterrows():
    p = pathlib.Path(r["path"])
    print(f"  File:     {r['filename']}")
    print(f"  Label:    {r['label']}")
    print(f"  Folder:   {r['folder']}")
    print(f"  Size:     {r['size_mb']} MB")
    print(f"  Path:     {r['path']}")
    print(f"  Exists:   {p.exists()}")
    print(f"  Error:    {r['decode_error']}")

    # Try opening with OpenCV to confirm
    cap = cv2.VideoCapture(str(p))
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else None
    ret, frame = cap.read()
    decoded = ret and frame is not None
    cap.release()
    print(f"  CV2 open: {opened}  decode_first_frame: {decoded}")
    if opened and fps:
        print(f"  CV2 fps:  {fps}")
    print()

# Count by label
print("=" * 80)
print("Summary by label:")
for label in ["normal", "abnormal", "unknown"]:
    count = len(bad[bad["label"] == label])
    print(f"  {label}: {count}")
