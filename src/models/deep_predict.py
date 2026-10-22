"""
src/models/deep_predict.py
---------------------------
Single-video inference using the trained CNN-LSTM model.
API mirrors src/models/predict.py so the dashboard can call either model.
"""

import torch
import numpy as np
from pathlib import Path

from src.models.deep_model import load_deep_model, LABEL_NAMES, FRAME_TRANSFORM
from src.models.deep_dataset import _extract_frame_features
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path

logger = get_logger(__name__)


def predict_video_deep(
    video_path: Path,
    cfg: dict | None = None,
    model=None,
) -> dict:
    """
    Run CNN-LSTM inference on a single video.

    Returns the same dict schema as predict_video() in predict.py so
    the dashboard can use either model interchangeably.
    """
    result = {
        "label":          "error",
        "label_id":       -1,
        "score_abnormal": 0.0,
        "score_normal":   0.0,
        "model_loaded":   False,
        "feature_ok":     False,
        "error":          None,
        "model_type":     "cnn_lstm",
    }

    if cfg is None:
        cfg = load_config()

    if model is None:
        ckpt_path = (
            get_path(cfg, "training", "model_output_dir")
            / cfg["training"].get("deep_checkpoint_name", "deep_model.pth")
        )
        if not ckpt_path.exists():
            result["error"] = (
                f"CNN-LSTM checkpoint not found at {ckpt_path}.  "
                "Run scripts/train_deep_model.py first."
            )
            return result
        try:
            model = load_deep_model(ckpt_path)
            result["model_loaded"] = True
        except Exception as exc:
            result["error"] = f"Failed to load deep model: {exc}"
            return result
    else:
        result["model_loaded"] = True

    model.eval()
    device = torch.device("cpu")
    video_path = Path(video_path)

    if not video_path.exists():
        result["error"] = f"Video file not found: {video_path}"
        return result

    interval  = cfg["preprocessing"]["frame_sample_interval"]
    max_frames = 150

    features = _extract_frame_features(video_path, model, interval, max_frames, device)
    if features is None:
        result["error"] = "Feature extraction failed (video unreadable or too short)."
        return result

    result["feature_ok"] = True

    try:
        # Add batch dimension, create length tensor
        x       = features.unsqueeze(0)                         # (1, seq, 576)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x, lengths)                          # (1, 2)
            probs  = torch.softmax(logits, dim=1).squeeze(0)    # (2,)

        label_id = int(torch.argmax(probs).item())
        result["label"]          = LABEL_NAMES.get(label_id, "unknown")
        result["label_id"]       = label_id
        result["score_normal"]   = round(float(probs[0].item()), 4)
        result["score_abnormal"] = round(float(probs[1].item()), 4)
    except Exception as exc:
        result["error"] = f"Prediction failed: {exc}"
        logger.error(result["error"])

    return result
