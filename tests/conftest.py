"""
tests/conftest.py
-----------------
Pytest fixtures shared across all test modules.
All tests use synthetic data — no original dataset files are touched.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import cv2
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg(project_root):
    """Load the real project config."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.utils.paths import load_config
    return load_config()


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """
    Create a tiny synthetic AVI video (20 frames, 64x64, 10 fps).
    Returns the path.  The file is written once per test session.
    """
    tmp = tmp_path_factory.mktemp("videos")
    vid_path = tmp / "synthetic.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))

    rng = np.random.default_rng(42)
    for i in range(20):
        frame = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()

    assert vid_path.exists(), "Synthetic video was not created"
    return vid_path


@pytest.fixture(scope="session")
def synthetic_manifest():
    """
    Create a small in-memory manifest DataFrame with 6 decodable labelled videos.
    Three normal, three abnormal.
    """
    rows = [
        {"filename": "norm1.avi", "folder": "Normal",   "label": "normal",
         "source_group": "labelled", "decoded": True, "split": "train",
         "path": "/fake/Normal/norm1.avi",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".avi",
         "readable": True, "decode_error": None},
        {"filename": "norm2.avi", "folder": "Normal",   "label": "normal",
         "source_group": "labelled", "decoded": True, "split": "train",
         "path": "/fake/Normal/norm2.avi",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".avi",
         "readable": True, "decode_error": None},
        {"filename": "norm3.avi", "folder": "Normal",   "label": "normal",
         "source_group": "labelled", "decoded": True, "split": "test",
         "path": "/fake/Normal/norm3.avi",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".avi",
         "readable": True, "decode_error": None},
        {"filename": "abn1.avi", "folder": "Abnormal",  "label": "abnormal",
         "source_group": "labelled", "decoded": True, "split": "train",
         "path": "/fake/Abnormal/abn1.avi",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".avi",
         "readable": True, "decode_error": None},
        {"filename": "abn2.avi", "folder": "Abnormal",  "label": "abnormal",
         "source_group": "labelled", "decoded": True, "split": "train",
         "path": "/fake/Abnormal/abn2.avi",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".avi",
         "readable": True, "decode_error": None},
        {"filename": "abn3.avi", "folder": "Abnormal",  "label": "abnormal",
         "source_group": "labelled", "decoded": True, "split": "test",
         "path": "/fake/Abnormal/abn3.avi",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".avi",
         "readable": True, "decode_error": None},
        {"filename": "extra1.mov", "folder": "Extra",   "label": "unknown",
         "source_group": "extra", "decoded": True, "split": float("nan"),
         "path": "/fake/Extra/extra1.mov",
         "fps": 25.0, "width": 480, "height": 360, "frame_count": 100,
         "duration_s": 4.0, "size_mb": 1.0, "extension": ".mov",
         "readable": True, "decode_error": None},
    ]
    return pd.DataFrame(rows)
