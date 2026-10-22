"""Inspect potential related clips for grouping before split."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "dataset/dataset_manifest.csv")
lab = df[(df["source_group"] == "labelled") & (df["decoded"] == True)]

print("Labelled decodable videos (20 total):")
print(lab[["filename", "label", "folder", "duration_s", "frame_count", "fps", "width", "height"]].to_string(index=False))
print()

# Check for related clips by prefix
print("Potential related clips (same filename prefix):")
for prefix in ["263C044", "CRW", "data", "flash", "8", "13"]:
    matches = lab[lab["filename"].str.startswith(prefix)]
    if len(matches) > 1:
        print(f"  Prefix '{prefix}': {matches['filename'].tolist()}")
        for _, r in matches.iterrows():
            print(f"    {r['filename']:35s}  label={r['label']}  "
                  f"dur={r['duration_s']}s  frames={r['frame_count']}  "
                  f"fps={r['fps']}  {r['width']}x{r['height']}")

print()
# Check 263C044_060_c and 263C044_064_c specifically
print("Detailed inspection of 263C044 pair:")
pair = lab[lab["filename"].str.contains("263C044")]
for _, r in pair.iterrows():
    print(f"  {r['filename']}: {r['width']}x{r['height']} @ {r['fps']}fps, "
          f"{r['frame_count']} frames, {r['duration_s']}s, label={r['label']}")
print()
print("Evidence for same recording session:")
print("  - Same prefix '263C044' suggests same camera/source")
print("  - Both are 1920x1080 @ 23.976fps")
print("  - Both labelled abnormal")
print("  - Different suffixes (_060_c vs _064_c) suggest different segments/clips")
print("  - Frame counts differ (271 vs 483) — different durations")
print("  - Conclusion: LIKELY related clips from same event — should be grouped")
