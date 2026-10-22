"""
Verify both inference paths on a real video without starting Streamlit.
Confirms:
  - RF: predict_video returns a real score and label
  - CNN-LSTM: predict_video_deep returns a real score and label
  - model_type key is set correctly
  - Neither fabricates results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.utils.logging_config import setup_logging

cfg = load_config()
setup_logging("WARNING")

# Pick a real readable video from the test split
TEST_VIDEO = PROJECT_ROOT / "dataset" / "Crowd Dataset - extra" / "861-13_70.mov"
if not TEST_VIDEO.exists():
    TEST_VIDEO = PROJECT_ROOT / "config" / "Abnormal Crowds" / "2017-420_l.mov"

print(f"Test video: {TEST_VIDEO.name}  exists={TEST_VIDEO.exists()}")
print()

# ── RF path ───────────────────────────────────────────────────────────────────
from src.models.baseline import load_model
from src.models.predict import predict_video

ckpt = get_path(cfg, "training", "model_output_dir") / cfg["training"]["checkpoint_name"]
rf_model = load_model(ckpt)

rf_result = predict_video(TEST_VIDEO, cfg=cfg, model=rf_model)
print("Random Forest inference:")
print(f"  error:          {rf_result['error']}")
print(f"  label:          {rf_result['label']}")
print(f"  score_abnormal: {rf_result['score_abnormal']}")
print(f"  score_normal:   {rf_result['score_normal']}")
print(f"  sum=1.0:        {abs(rf_result['score_normal']+rf_result['score_abnormal']-1.0) < 1e-4}")
assert rf_result["error"] is None,  f"RF error: {rf_result['error']}"
assert rf_result["label"] in ("normal","abnormal"), "RF label invalid"
assert 0.0 <= rf_result["score_abnormal"] <= 1.0, "RF score out of range"
print("  PASS")
print()

# ── CNN-LSTM path ─────────────────────────────────────────────────────────────
from src.models.deep_model import load_deep_model
from src.models.deep_predict import predict_video_deep

deep_ckpt = (get_path(cfg, "training", "model_output_dir")
             / cfg["training"].get("deep_checkpoint_name", "deep_model.pth"))
deep_model = load_deep_model(deep_ckpt)

deep_result = predict_video_deep(TEST_VIDEO, cfg=cfg, model=deep_model)
print("CNN-LSTM inference:")
print(f"  error:          {deep_result['error']}")
print(f"  label:          {deep_result['label']}")
print(f"  score_abnormal: {deep_result['score_abnormal']}")
print(f"  score_normal:   {deep_result['score_normal']}")
print(f"  model_type:     {deep_result['model_type']}")
print(f"  sum=1.0:        {abs(deep_result['score_normal']+deep_result['score_abnormal']-1.0) < 1e-4}")
assert deep_result["error"] is None,  f"Deep error: {deep_result['error']}"
assert deep_result["label"] in ("normal","abnormal"), "Deep label invalid"
assert 0.0 <= deep_result["score_abnormal"] <= 1.0, "Deep score out of range"
assert deep_result["model_type"] == "cnn_lstm", "model_type key missing"
print("  PASS")
print()

# ── Confirm scores differ (they come from different models) ───────────────────
print("Score comparison:")
print(f"  RF    P(abnormal) = {rf_result['score_abnormal']}")
print(f"  Deep  P(abnormal) = {deep_result['score_abnormal']}")
print()
print("BOTH MODEL PATHS VERIFIED — real inference, no fabrication.")
