"""
src/data/extract_frames.py
--------------------------
Extracts frames from videos at a configurable sampling interval.
Frames are stored under generated_data/frames/<label>/<video_stem>/
This module NEVER modifies the original video files.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator

from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path, ensure_dir

logger = get_logger(__name__)


def resize_frame(frame: np.ndarray, width: int, height: int,
                 preserve_aspect: bool = True) -> np.ndarray:
    """
    Resize a BGR frame to (width, height).
    If preserve_aspect is True, the frame is scaled so its longest side
    fits the target, then letter-/pillar-boxed to fill exactly.
    """
    if not preserve_aspect:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    h0, w0 = frame.shape[:2]
    scale = min(width / w0, height / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad with black to reach target size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_off = (height - new_h) // 2
    x_off = (width  - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    sample_interval: int = 10,
    target_width: int = 224,
    target_height: int = 224,
    preserve_aspect: bool = True,
) -> dict:
    """
    Extract sampled frames from one video.

    Parameters
    ----------
    video_path       : Path to the source video (not modified).
    output_dir       : Directory to write extracted frames (created if needed).
    sample_interval  : Save every Nth frame (1 = every frame).
    target_width/height : Output frame dimensions.
    preserve_aspect  : Whether to letterbox instead of distort.

    Returns
    -------
    dict with keys: success, frames_extracted, error
    """
    result = {"success": False, "frames_extracted": 0, "error": None}

    ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        result["error"] = "cv2_cannot_open"
        return result

    frame_idx  = 0
    saved      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            resized = resize_frame(frame, target_width, target_height, preserve_aspect)
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), resized)
            saved += 1

        frame_idx += 1

    cap.release()

    if saved == 0:
        result["error"] = "no_frames_extracted"
    else:
        result["success"] = True
        result["frames_extracted"] = saved

    return result


def stream_frames(
    video_path: Path,
    sample_interval: int = 10,
    target_width: int = 224,
    target_height: int = 224,
    preserve_aspect: bool = True,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, resized_frame) for every sampled frame without
    saving to disk.  Used by feature extraction and inference pipelines.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video for streaming: %s", video_path.name)
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            resized = resize_frame(frame, target_width, target_height, preserve_aspect)
            yield idx, resized
        idx += 1

    cap.release()


def extract_all_frames(
    manifest: pd.DataFrame,
    cfg: dict | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Extract frames for every decodable labelled video in the manifest.

    Parameters
    ----------
    manifest  : The inventory DataFrame (from src.data.inventory).
    cfg       : Config dict.
    overwrite : If False, skip videos whose output folder already exists.

    Returns
    -------
    DataFrame with columns: filename, frames_extracted, success, error
    """
    if cfg is None:
        from src.utils.paths import load_config
        cfg = load_config()

    interval    = cfg["preprocessing"]["frame_sample_interval"]
    width       = cfg["preprocessing"]["target_width"]
    height      = cfg["preprocessing"]["target_height"]
    aspect      = cfg["preprocessing"]["preserve_aspect_ratio"]
    frames_root = get_path(cfg, "preprocessing", "frames_output_dir")

    ensure_dir(frames_root)

    # Only process labelled, decodable videos
    subset = manifest[
        (manifest["source_group"] == "labelled") &
        (manifest["decoded"] == True)
    ].copy()

    if subset.empty:
        logger.warning("No labelled decodable videos to extract frames from.")
        return pd.DataFrame()

    results = []
    for _, row in subset.iterrows():
        video_path = Path(row["path"])
        label      = row["label"]
        stem       = video_path.stem
        out_dir    = frames_root / label / stem

        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            existing = len(list(out_dir.glob("*.jpg")))
            logger.info("Skipping (already extracted %d frames): %s", existing, stem)
            results.append({
                "filename": row["filename"],
                "frames_extracted": existing,
                "success": True,
                "error": "skipped_already_exists",
            })
            continue

        logger.info("Extracting frames: %s → %s", video_path.name, out_dir)
        res = extract_frames_from_video(
            video_path=video_path,
            output_dir=out_dir,
            sample_interval=interval,
            target_width=width,
            target_height=height,
            preserve_aspect=aspect,
        )
        res["filename"] = row["filename"]
        results.append(res)
        if res["success"]:
            logger.info("  → %d frames saved", res["frames_extracted"])
        else:
            logger.warning("  → FAILED: %s", res["error"])

    return pd.DataFrame(results)
