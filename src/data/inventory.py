"""
src/data/inventory.py
----------------------
Discovers all video files under the dataset root, extracts metadata via
OpenCV, and returns a pandas DataFrame with one row per video.

Nothing in this module modifies the source dataset.
"""

import cv2
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_dataset_root

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".wmv", ".flv", ".mkv",
                    ".mpeg", ".mpg", ".m4v"}


def _probe_video(path: Path) -> dict:
    """
    Open a single video file with OpenCV and extract metadata.
    Returns a dict regardless of success; records error reason on failure.
    """
    record: dict = {
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "readable": False,
        "decoded": False,
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "duration_s": None,
        "decode_error": None,
    }

    try:
        cap = cv2.VideoCapture(str(path))
        record["readable"] = cap.isOpened()

        if record["readable"]:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            record["fps"]         = round(fps, 3) if fps > 0 else None
            record["width"]       = w if w > 0 else None
            record["height"]      = h if h > 0 else None
            record["frame_count"] = fc if fc > 0 else None

            if record["fps"] and record["frame_count"]:
                record["duration_s"] = round(record["frame_count"] / record["fps"], 2)

            # Attempt to decode the first frame to confirm the codec works
            ret, frame = cap.read()
            if ret and frame is not None:
                record["decoded"] = True
            else:
                record["decode_error"] = "first_frame_decode_failed (codec not supported)"

        else:
            record["decode_error"] = "cv2_cannot_open"

        cap.release()

    except Exception as exc:
        record["decode_error"] = str(exc)

    return record


def build_inventory(cfg: dict | None = None, extra_folder: str | None = None) -> pd.DataFrame:
    """
    Walk the dataset root and probe every video file.

    Parameters
    ----------
    cfg          : Loaded config dict.  If None, config.yaml is loaded automatically.
    extra_folder : Name of the 'extra' folder (unlabelled). Its videos get
                   label='unknown' and source_group='extra'.

    Returns
    -------
    pd.DataFrame with columns:
        path, filename, extension, folder, label, source_group,
        size_mb, readable, decoded, fps, width, height,
        frame_count, duration_s, decode_error
    """
    if cfg is None:
        cfg = load_config()

    dataset_root = get_dataset_root(cfg)
    normal_folder_cfg   = cfg["dataset"]["normal_folder"]
    abnormal_folder_cfg = cfg["dataset"]["abnormal_folder"]
    extra_folder_cfg    = extra_folder or cfg["dataset"]["extra_folder"]
    allowed_exts        = {e.lower() for e in cfg["dataset"]["video_extensions"]}

    # Build resolved absolute paths for each labelled folder.
    # The config value may be an absolute path or relative to dataset_root.
    def _resolve_folder(val: str) -> Path:
        p = Path(val)
        return p if p.is_absolute() else dataset_root / val

    normal_dir   = _resolve_folder(normal_folder_cfg)
    abnormal_dir = _resolve_folder(abnormal_folder_cfg)
    extra_dir    = _resolve_folder(extra_folder_cfg)

    logger.info("Scanning dataset root: %s", dataset_root)
    logger.info("  Normal    folder: %s (exists=%s)", normal_dir,   normal_dir.exists())
    logger.info("  Abnormal  folder: %s (exists=%s)", abnormal_dir, abnormal_dir.exists())
    logger.info("  Extra     folder: %s (exists=%s)", extra_dir,    extra_dir.exists())

    records = []
    for video_path in sorted(dataset_root.rglob("*")):
        if video_path.suffix.lower() not in allowed_exts:
            continue
        if not video_path.is_file():
            continue

        folder_name = video_path.parent.name
        video_parent_resolved = video_path.parent.resolve()

        # Assign label and source group based on which labelled folder the
        # video lives in (compare resolved absolute paths).
        if normal_dir.exists() and video_parent_resolved == normal_dir.resolve():
            label        = "normal"
            source_group = "labelled"
        elif abnormal_dir.exists() and video_parent_resolved == abnormal_dir.resolve():
            label        = "abnormal"
            source_group = "labelled"
        elif extra_dir.exists() and video_parent_resolved == extra_dir.resolve():
            label        = "unknown"
            source_group = "extra"
        else:
            label        = "unknown"
            source_group = "other"

        logger.debug("Probing: %s", video_path.name)
        record = _probe_video(video_path)
        record["folder"]       = folder_name
        record["label"]        = label
        record["source_group"] = source_group
        records.append(record)

    if not records:
        logger.warning("No video files found under %s", dataset_root)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Reorder columns for readability
    col_order = [
        "filename", "folder", "label", "source_group",
        "extension", "size_mb", "readable", "decoded",
        "fps", "width", "height", "frame_count", "duration_s",
        "decode_error", "path"
    ]
    df = df[[c for c in col_order if c in df.columns]]

    logger.info(
        "Inventory complete. Total=%d  Normal=%d  Abnormal=%d  Extra=%d  "
        "Decodable=%d  Non-decodable=%d",
        len(df),
        (df.label == "normal").sum(),
        (df.label == "abnormal").sum(),
        (df.label == "unknown").sum(),
        df.decoded.sum(),
        (~df.decoded).sum(),
    )
    return df


def save_inventory(df: pd.DataFrame, output_path: Path) -> None:
    """Save the inventory DataFrame to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Inventory saved to %s", output_path)


def load_inventory(manifest_path: Path) -> pd.DataFrame:
    """Load a previously saved manifest CSV."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run scripts/inspect_dataset.py first."
        )
    return pd.read_csv(manifest_path)
