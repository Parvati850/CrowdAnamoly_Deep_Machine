"""
tests/test_inventory.py
------------------------
Unit tests for dataset discovery and inventory building.
Uses synthetic data only — no original dataset files are touched.
"""

import pytest
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


def _make_video(path: Path, n_frames: int = 15, w: int = 64, h: int = 64) -> None:
    """Write a small readable synthetic AVI to path."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_inventory_finds_videos(tmp_path, project_root):
    """Inventory discovers videos placed under a fake dataset root."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.inventory import build_inventory

    # Create fake folder structure
    normal_dir   = tmp_path / "Normal"
    abnormal_dir = tmp_path / "Abnormal"
    normal_dir.mkdir()
    abnormal_dir.mkdir()

    _make_video(normal_dir / "n1.avi")
    _make_video(normal_dir / "n2.avi")
    _make_video(abnormal_dir / "a1.avi")

    fake_cfg = {
        "dataset": {
            "dataset_root":   str(tmp_path),
            "normal_folder":  "Normal",
            "abnormal_folder":"Abnormal",
            "extra_folder":   "Extra",
            "video_extensions": [".avi", ".mov", ".mp4"],
        }
    }

    df = build_inventory(fake_cfg)
    assert len(df) == 3, f"Expected 3 videos, got {len(df)}"
    assert set(df["label"].unique()) == {"normal", "abnormal"}
    assert (df["label"] == "normal").sum() == 2
    assert (df["label"] == "abnormal").sum() == 1


def test_inventory_marks_unreadable(tmp_path, project_root):
    """Inventory correctly marks unreadable files."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.inventory import build_inventory

    normal_dir = tmp_path / "Normal"
    normal_dir.mkdir()

    # Write a corrupt file
    bad_file = normal_dir / "corrupt.avi"
    bad_file.write_bytes(b"this is not a video file 0000")

    # Write a valid file
    _make_video(normal_dir / "good.avi")

    fake_cfg = {
        "dataset": {
            "dataset_root":   str(tmp_path),
            "normal_folder":  "Normal",
            "abnormal_folder":"Abnormal",
            "extra_folder":   "Extra",
            "video_extensions": [".avi"],
        }
    }

    df = build_inventory(fake_cfg)
    assert len(df) == 2
    decodable = df[df["decoded"] == True]
    assert len(decodable) == 1
    assert decodable.iloc[0]["filename"] == "good.avi"


def test_inventory_extra_labelled_unknown(tmp_path, project_root):
    """Videos in the extra folder get label='unknown'."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.inventory import build_inventory

    extra_dir = tmp_path / "Extra"
    extra_dir.mkdir()
    _make_video(extra_dir / "e1.avi")

    fake_cfg = {
        "dataset": {
            "dataset_root":   str(tmp_path),
            "normal_folder":  "Normal",
            "abnormal_folder":"Abnormal",
            "extra_folder":   "Extra",
            "video_extensions": [".avi"],
        }
    }

    df = build_inventory(fake_cfg)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "unknown"
    assert df.iloc[0]["source_group"] == "extra"


def test_save_and_reload_inventory(tmp_path, project_root):
    """Inventory can be saved to CSV and reloaded."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.inventory import build_inventory, save_inventory, load_inventory

    d = tmp_path / "Normal"
    d.mkdir()
    _make_video(d / "n.avi")

    fake_cfg = {
        "dataset": {
            "dataset_root":   str(tmp_path),
            "normal_folder":  "Normal",
            "abnormal_folder":"Abnormal",
            "extra_folder":   "Extra",
            "video_extensions": [".avi"],
        }
    }

    df = build_inventory(fake_cfg)
    out = tmp_path / "manifest.csv"
    save_inventory(df, out)
    df2 = load_inventory(out)
    assert len(df2) == len(df)
    assert list(df2.columns) == list(df.columns)
