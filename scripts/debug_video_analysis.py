"""
scripts/debug_video_analysis.py
--------------------------------
Simulates the complete Video Analysis page workflow from app.py
using the actual test video (861-13_70.mov from extra dataset).
Exercises every step: upload → metadata → feature extraction →
model prediction → alert logic → result display.

Run:
    .\.venv\Scripts\python.exe scripts/debug_video_analysis.py
or:
    python scripts/debug_video_analysis.py
"""

import sys, shutil, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.utils.logging_config import setup_logging, get_logger

cfg = load_config()
setup_logging("INFO")
logger = get_logger("debug_video_analysis")

# ── Locate test video ────────────────────────────────────────────────────────
TEST_VIDEO = (
    PROJECT_ROOT / "dataset" / "Crowd Dataset - extra" / "861-13_70.mov"
)
if not TEST_VIDEO.exists():
    # Try a labelled video as fallback
    TEST_VIDEO = (
        PROJECT_ROOT / "config" / "Abnormal Crowds" / "2017-420_l.mov"
    )

if not TEST_VIDEO.exists():
    logger.error("No test video found. Checked: %s", TEST_VIDEO)
    sys.exit(1)

logger.info("Test video: %s  (%.2f MB)", TEST_VIDEO.name,
            TEST_VIDEO.stat().st_size / 1024 / 1024)

# ── Step 1: Copy to temp file (simulates st.file_uploader write) ─────────────
suffix = TEST_VIDEO.suffix
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
shutil.copy2(TEST_VIDEO, tmp.name)
tmp_path = Path(tmp.name)
tmp.close()
logger.info("Step 1 PASS – temp file written: %s", tmp_path)

# ── Step 2: Probe metadata ────────────────────────────────────────────────────
import cv2

def probe(path):
    info = {}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info = dict(fps=round(fps,2), width=w, height=h, frame_count=fc)
        ret, frame = cap.read()
        info["decodable"] = bool(ret and frame is not None)
    finally:
        cap.release()
    return info

meta = probe(tmp_path)
assert meta and meta["decodable"], f"Video not decodable: {meta}"
logger.info("Step 2 PASS – metadata: %s", meta)

# ── Step 3: Instantiate AlertManager with source_name ────────────────────────
from src.inference.alerts import AlertManager

alert_mgr = AlertManager(
    threshold=cfg["alerts"]["threshold"],
    persistence=cfg["alerts"]["persistence"],
    cooldown_seconds=cfg["alerts"]["cooldown_seconds"],
    source_name=TEST_VIDEO.name,
)
logger.info("Step 3 PASS – AlertManager instantiated, source_name='%s'",
            alert_mgr.source_name)

# ── Step 4: Load model ────────────────────────────────────────────────────────
from src.models.baseline import load_model

ckpt = get_path(cfg, "training", "model_output_dir") / cfg["training"]["checkpoint_name"]
assert ckpt.exists(), f"Model checkpoint missing: {ckpt}"
model = load_model(ckpt)
logger.info("Step 4 PASS – model loaded: %s", type(model).__name__)

# ── Step 5: Run analyse_video ─────────────────────────────────────────────────
from src.inference.video_inference import analyse_video

result = analyse_video(
    video_path=tmp_path,
    cfg=cfg,
    model=model,
    alert_manager=alert_mgr,
    window_size=20,
)

assert result["error"] is None, f"analyse_video error: {result['error']}"
logger.info("Step 5 PASS – analysis complete")
logger.info("  Windows analysed:  %d", len(result["windows"]))
logger.info("  Overall score:     %s", result["overall_score"])
logger.info("  Overall label:     %s", result["overall_label"])
logger.info("  Processing FPS:    %.1f", result["fps_processing"])
logger.info("  Alerts fired:      %d", alert_mgr.active_count)

# ── Step 6: Verify feature vector shape ──────────────────────────────────────
from src.features.motion_features import extract_video_features, feature_names
feat = extract_video_features(
    tmp_path,
    sample_interval=cfg["preprocessing"]["frame_sample_interval"],
    target_width=cfg["preprocessing"]["target_width"],
    target_height=cfg["preprocessing"]["target_height"],
)
assert feat is not None, "Feature extraction returned None"
assert len(feat) == 33, f"Expected 33 features, got {len(feat)}"
assert len(feat) == len(feature_names()), "feature_names() length mismatch"
logger.info("Step 6 PASS – feature vector shape: %d", len(feat))

# ── Step 7: Direct predict_video call ────────────────────────────────────────
from src.models.predict import predict_video

pred = predict_video(TEST_VIDEO, cfg=cfg, model=model)
assert pred["error"] is None, f"predict_video error: {pred['error']}"
assert pred["label"] in ("normal", "abnormal")
assert 0.0 <= pred["score_abnormal"] <= 1.0
assert abs(pred["score_normal"] + pred["score_abnormal"] - 1.0) < 1e-4
logger.info("Step 7 PASS – prediction: label=%s  score=%.4f",
            pred["label"], pred["score_abnormal"])

# ── Step 8: Threshold consistency check ──────────────────────────────────────
threshold = cfg["alerts"]["threshold"]
score     = pred["score_abnormal"]
expected_flag = score >= threshold
logger.info(
    "Step 8 PASS – threshold check: score=%.4f  threshold=%.2f  "
    "flagged=%s", score, threshold, expected_flag
)

# ── Step 9: Alert dedup check ─────────────────────────────────────────────────
event_ids = [e.event_id for e in alert_mgr.events]
assert len(event_ids) == len(set(event_ids)), "Duplicate event_ids found!"
logger.info("Step 9 PASS – no duplicate alert IDs (%d total)", len(event_ids))

# ── Step 10: Cleanup temp file ────────────────────────────────────────────────
try:
    tmp_path.unlink()
    logger.info("Step 10 PASS – temp file cleaned up")
except Exception as e:
    logger.warning("Temp file cleanup failed (non-critical): %s", e)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  ALL STEPS PASSED")
print("=" * 60)
print(f"  Video:          {TEST_VIDEO.name}")
print(f"  Resolution:     {meta['width']}x{meta['height']} @ {meta['fps']} fps")
print(f"  Frame count:    {meta['frame_count']}")
print(f"  Features:       {len(feat)} values")
print(f"  Windows:        {len(result['windows'])}")
print(f"  Overall score:  {result['overall_score']}")
print(f"  Overall label:  {result['overall_label']}")
print(f"  Threshold:      {threshold}")
print(f"  Alerts fired:   {alert_mgr.active_count}")
print(f"  source_name:    {alert_mgr.source_name}")
print("=" * 60)
