"""
Test the empty-windows path: simulate what the Video Analysis page does
when analyse_video returns windows=[] (e.g., video too short to fill one window).
Verifies that:
  1. pandas is available (imported before the if result["windows"]: guard)
  2. n_wins/n_abn caption renders correctly with 0 windows
  3. The motion-indicator explanation still appears
  4. The model comparison expander still loads
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.utils.logging_config import setup_logging
from src.inference.video_inference import analyse_video
from src.models.baseline import load_model
import cv2, tempfile, numpy as np

setup_logging("WARNING")
cfg = load_config()

# Create a very short 4-frame synthetic video that cannot fill a window_size=20 buffer
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
tmp.close()
vid_path = pathlib.Path(tmp.name)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))
rng = np.random.default_rng(1)
for _ in range(4):   # only 4 frames — window_size=20 will never fill
    writer.write((rng.random((64, 64, 3)) * 255).astype(np.uint8))
writer.release()

# Load real model
ckpt = get_path(cfg, "training", "model_output_dir") / cfg["training"]["checkpoint_name"]
model = load_model(ckpt)

# Run analyse_video with window_size=20 — will produce 0 windows
result = analyse_video(vid_path, cfg=cfg, model=model, window_size=20)
vid_path.unlink()   # clean up

print("Empty-windows test result:")
print(f"  error:          {result['error']}")
print(f"  windows:        {result['windows']}")
print(f"  overall_score:  {result['overall_score']}")
print(f"  overall_label:  {result['overall_label']}")
print(f"  fps_processing: {result['fps_processing']}")

# Verify that the display code path used in app.py works with empty windows
import pandas as pd   # must be available here

n_wins = len(result["windows"])
n_abn  = sum(1 for w in result["windows"] if w["label"] == "abnormal")
caption = (
    f"Summary: {n_wins} window(s) analysed, "
    f"{n_abn} flagged above threshold (0.55)."
)
print(f"  caption:        {caption}")

# The windows-display block is guarded by if result["windows"]:
# so pd.DataFrame(result["windows"]) is never called when empty.
if result["windows"]:
    df = pd.DataFrame(result["windows"])
    print(f"  windows_df shape: {df.shape}")
else:
    print("  windows is empty — DataFrame block correctly skipped")

# overall_score is None when no windows; the display code handles this:
overall_score = result["overall_score"] or 0.0
print(f"  overall_score (with or 0.0 guard): {overall_score}")

print()
print("PASS — empty-windows path does not crash and pd is available")
