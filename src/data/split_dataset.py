"""
src/data/split_dataset.py
--------------------------
Creates train / validation / test splits at the VIDEO level to prevent
data leakage.  All frames from the same video stay in one split.

Related clips from the same recording session are grouped together
so they cannot be split across train and test (group-aware splitting).

Strategy:
  - Use only labelled, decodable videos.
  - Assign a group_id to each video based on prefix analysis.
  - Split by GROUP, not by individual video, so related clips stay together.
  - Stratify by label where possible.
  - If ≥ 4 videos per class, do train/val/test.
  - If fewer, fall back to train/test only (and document the limitation).
  - Never split by frame; the video (or group) is the unit of analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Clips identified as related (same recording session) based on
# evidence: same filename prefix, same resolution, same fps, same label.
# Only groups with strong evidence are listed here — not just similar names.
RELATED_CLIP_GROUPS = [
    ["263C044_060_c.mov", "263C044_064_c.mov"],
]


def _assign_group_ids(labelled: pd.DataFrame) -> pd.Series:
    """
    Assign a group_id to each video.  Videos in the same group
    share a group_id so they cannot be split across train/test.

    Default: each video is its own group (group_id = filename).
    Related clips get the same group_id based on RELATED_CLIP_GROUPS.
    """
    group_map = {}
    for group in RELATED_CLIP_GROUPS:
        group_name = "GROUP_" + group[0].rsplit("_", 2)[0]  # e.g. GROUP_263C044
        for fn in group:
            group_map[fn] = group_name

    # Default: each video is its own group
    result = labelled["filename"].copy()
    for idx, row in labelled.iterrows():
        fn = row["filename"]
        if fn in group_map:
            result.loc[idx] = group_map[fn]
        else:
            result.loc[idx] = fn  # unique group = itself
    return result


def split_manifest(
    manifest: pd.DataFrame,
    test_size: float = 0.25,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Assign a 'split' column ('train', 'val', 'test') to each labelled,
    decodable video in the manifest.

    Parameters
    ----------
    manifest    : Full inventory DataFrame.
    test_size   : Fraction of labelled videos for the test set.
    val_size    : Fraction of remaining (non-test) videos for validation.
    random_seed : Controls reproducibility.

    Returns
    -------
    The manifest DataFrame with an added 'split' column.
    NaN for unlabelled / non-decodable rows.
    """
    df = manifest.copy()
    df["split"] = pd.NA
    df["group_id"] = pd.NA

    labelled = df[
        (df["source_group"] == "labelled") &
        (df["decoded"] == True)
    ].copy()

    if labelled.empty:
        logger.error("No labelled decodable videos – cannot split dataset.")
        return df

    # Assign group IDs for related-clip grouping
    labelled["group_id"] = _assign_group_ids(labelled)
    df.loc[labelled.index, "group_id"] = labelled["group_id"]

    min_per_class = labelled.groupby("label").size().min()
    n_total = len(labelled)
    n_groups = labelled["group_id"].nunique()
    logger.info("Split: %d videos in %d groups", n_total, n_groups)

    # ── Determine strategy ──────────────────────────────────────────────────
    if min_per_class < 2:
        logger.warning(
            "Fewer than 2 samples in at least one class. "
            "All labelled videos assigned to 'train'. Evaluation will not be valid."
        )
        df.loc[labelled.index, "split"] = "train"
        return df

    # ── Group-aware stratified split ────────────────────────────────────────
    # We split by GROUP, not by individual video, so related clips
    # stay in the same split.  We use the group's dominant label for
    # stratification.

    # Build a group-level dataframe: one row per group
    group_labels = labelled.groupby("group_id")["label"].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    group_indices = labelled.groupby("group_id").apply(
        lambda x: list(x.index), include_groups=False
    )

    group_names = group_labels.index.tolist()
    group_label_list = group_labels.values
    group_idx_map = dict(zip(group_names, group_indices.values))

    # Split groups into train+val vs test
    try:
        train_val_groups, test_groups = train_test_split(
            group_names,
            test_size=test_size,
            stratify=group_label_list,
            random_state=random_seed,
        )
    except ValueError as exc:
        logger.warning("Stratified group test split failed (%s). Using random split.", exc)
        train_val_groups, test_groups = train_test_split(
            group_names,
            test_size=test_size,
            random_state=random_seed,
        )

    # Assign test split
    for g in test_groups:
        for idx in group_idx_map[g]:
            df.loc[idx, "split"] = "test"

    # Split train+val groups into train and val
    train_val_label_list = [group_labels[g] for g in train_val_groups]
    if len(train_val_groups) >= 4 and min(pd.Series(train_val_label_list).value_counts()) >= 2:
        try:
            train_groups, val_groups = train_test_split(
                train_val_groups,
                test_size=val_size / (1.0 - test_size),
                stratify=train_val_label_list,
                random_state=random_seed,
            )
        except ValueError:
            train_groups = train_val_groups
            val_groups = []
            logger.warning("Validation group split failed. Val set omitted.")
    else:
        train_groups = train_val_groups
        val_groups = []
        logger.warning(
            "Not enough groups for validation split (%d remaining after test). "
            "Val set omitted.",
            len(train_val_groups),
        )

    for g in train_groups:
        for idx in group_idx_map[g]:
            df.loc[idx, "split"] = "train"
    for g in val_groups:
        for idx in group_idx_map[g]:
            df.loc[idx, "split"] = "val"

    # Log split summary
    split_counts = df[df["split"].notna()].groupby(["split", "label"]).size()
    logger.info("Dataset split:\n%s", split_counts.to_string())

    # Verify no video appears in more than one split (sanity check)
    split_df = df[df["split"].notna()]
    for path_val in split_df["path"].unique():
        rows = split_df[split_df["path"] == path_val]
        if rows["split"].nunique() > 1:
            logger.error("DATA LEAKAGE: video %s appears in multiple splits!", path_val)

    # Verify no GROUP crosses splits
    grouped_split = split_df.groupby("group_id")["split"].nunique()
    leaking_groups = grouped_split[grouped_split > 1]
    if not leaking_groups.empty:
        logger.error("GROUP LEAKAGE: groups in multiple splits: %s", leaking_groups.index.tolist())
    else:
        logger.info("Group leakage check: PASS (no group crosses splits)")

    return df


def print_split_summary(df: pd.DataFrame) -> None:
    """Print a human-readable split summary."""
    split_df = df[df["split"].notna()]
    if split_df.empty:
        print("No split assignments found.")
        return

    print("\n── Dataset Split Summary ────────────────────────────────────────────")
    summary = split_df.groupby(["split", "label"]).size().rename("count").reset_index()
    for split in ["train", "val", "test"]:
        rows = summary[summary["split"] == split]
        if rows.empty:
            continue
        parts = ", ".join(f"{r['label']}={r['count']}" for _, r in rows.iterrows())
        print(f"  {split:<6} → {rows['count'].sum()} videos  ({parts})")
    print()
