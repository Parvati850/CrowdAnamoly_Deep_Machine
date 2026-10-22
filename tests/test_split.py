"""
tests/test_split.py
--------------------
Tests for dataset splitting — verifies no data leakage.
"""

import pytest
import pandas as pd
import numpy as np


def test_split_assigns_all_labelled(synthetic_manifest, project_root):
    """All labelled decodable videos get a split assignment."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.split_dataset import split_manifest

    # Use the fixture manifest which already has splits set
    df = synthetic_manifest.copy()
    # Reset splits to test the function itself
    df["split"] = pd.NA

    result = split_manifest(df, test_size=0.33, val_size=0.0, random_seed=42)

    labelled_dec = result[
        (result["source_group"] == "labelled") &
        (result["decoded"] == True)
    ]
    assert labelled_dec["split"].notna().all(), (
        "Not all labelled decodable videos got a split assignment"
    )


def test_no_video_in_multiple_splits(synthetic_manifest, project_root):
    """Each video appears in exactly one split."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.split_dataset import split_manifest

    df = synthetic_manifest.copy()
    df["split"] = pd.NA
    result = split_manifest(df, test_size=0.33, random_seed=42)

    split_df = result[result["split"].notna()]
    for path_val in split_df["path"].unique():
        splits = split_df[split_df["path"] == path_val]["split"].unique()
        assert len(splits) == 1, f"Video {path_val} appears in splits: {splits}"


def test_extra_videos_not_split(synthetic_manifest, project_root):
    """Extra (unlabelled) videos never get a split assignment."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.split_dataset import split_manifest

    df = synthetic_manifest.copy()
    df["split"] = pd.NA
    result = split_manifest(df, test_size=0.33, random_seed=42)

    extra_with_split = result[
        (result["source_group"] == "extra") &
        (result["split"].notna())
    ]
    assert len(extra_with_split) == 0, (
        f"Extra videos should not be split, found: {extra_with_split}"
    )


def test_stratification_preserves_both_classes(synthetic_manifest, project_root):
    """Both classes appear in the training split."""
    import sys
    sys.path.insert(0, str(project_root))
    from src.data.split_dataset import split_manifest

    df = synthetic_manifest.copy()
    df["split"] = pd.NA
    result = split_manifest(df, test_size=0.33, random_seed=42)

    train = result[result["split"] == "train"]
    labels_in_train = set(train["label"].unique())
    assert "normal"   in labels_in_train, "No normal videos in train split"
    assert "abnormal" in labels_in_train, "No abnormal videos in train split"
