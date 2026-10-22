"""
src/data/validate.py
---------------------
Validates a dataset manifest for quality, balance, and integrity issues.
Reports problems as structured findings without modifying anything.
"""

import pandas as pd
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def validate_manifest(df: pd.DataFrame) -> dict:
    """
    Run a suite of checks on the inventory DataFrame.

    Returns
    -------
    dict with keys:
        summary    – high-level counts
        warnings   – list of warning strings
        errors     – list of error strings
        label_dist – dict {label: count}
        decodable  – dict {label: decodable_count}
    """
    findings: dict = {
        "summary":    {},
        "warnings":   [],
        "errors":     [],
        "label_dist": {},
        "decodable":  {},
    }

    if df.empty:
        findings["errors"].append("Manifest is empty – no videos discovered.")
        return findings

    total        = len(df)
    label_counts = df["label"].value_counts().to_dict()
    decoded_per_label = (
        df[df["decoded"] == True]["label"].value_counts().to_dict()
    )

    findings["label_dist"] = label_counts
    findings["decodable"]  = decoded_per_label

    findings["summary"] = {
        "total_videos":           total,
        "labelled_videos":        int((df["source_group"] == "labelled").sum()),
        "extra_videos":           int((df["source_group"] == "extra").sum()),
        "fully_decodable":        int(df["decoded"].sum()),
        "non_decodable":          int((~df["decoded"]).sum()),
        "normal_videos":          int((df["label"] == "normal").sum()),
        "abnormal_videos":        int((df["label"] == "abnormal").sum()),
        "unknown_label_videos":   int((df["label"] == "unknown").sum()),
    }

    # ── Checks ──────────────────────────────────────────────────────────────

    # 1. Non-decodable videos
    bad = df[~df["decoded"]]
    if not bad.empty:
        for _, row in bad.iterrows():
            findings["warnings"].append(
                f"Cannot decode: {row['filename']} | Error: {row['decode_error']}"
            )

    # 2. Class imbalance
    n_normal   = findings["summary"]["normal_videos"]
    n_abnormal = findings["summary"]["abnormal_videos"]

    if n_normal == 0:
        findings["errors"].append("No NORMAL videos found. Check folder name in config.")
    if n_abnormal == 0:
        findings["errors"].append("No ABNORMAL videos found. Check folder name in config.")

    if n_normal > 0 and n_abnormal > 0:
        ratio = max(n_normal, n_abnormal) / min(n_normal, n_abnormal)
        if ratio >= 2:
            findings["warnings"].append(
                f"Class imbalance: normal={n_normal}, abnormal={n_abnormal} "
                f"(ratio {ratio:.1f}x). Consider class weighting."
            )

    # 3. Very small usable dataset
    labelled_decodable = (
        df[(df["source_group"] == "labelled") & (df["decoded"] == True)]
    )
    if len(labelled_decodable) < 10:
        findings["warnings"].append(
            f"Only {len(labelled_decodable)} labelled decodable videos available. "
            "This is very small for reliable supervised learning."
        )

    # 4. Potential duplicate filenames across folders
    dupes = df[df.duplicated(subset="filename", keep=False)]
    if not dupes.empty:
        findings["warnings"].append(
            f"Duplicate filenames across folders: {dupes['filename'].tolist()}"
        )

    # 5. Very short videos (< 2 s)
    short = df[(df["duration_s"].notna()) & (df["duration_s"] < 2.0)]
    if not short.empty:
        for _, row in short.iterrows():
            findings["warnings"].append(
                f"Very short video ({row['duration_s']} s): {row['filename']}"
            )

    # 6. Missing duration (frame count or fps missing)
    no_dur = df[df["duration_s"].isna() & df["decoded"]]
    if not no_dur.empty:
        for _, row in no_dur.iterrows():
            findings["warnings"].append(
                f"Duration could not be computed: {row['filename']} "
                f"(fps={row['fps']}, frame_count={row['frame_count']})"
            )

    # Log summary
    logger.info("Validation: %s", findings["summary"])
    for w in findings["warnings"]:
        logger.warning(w)
    for e in findings["errors"]:
        logger.error(e)

    return findings


def print_validation_report(findings: dict) -> None:
    """Pretty-print validation findings to stdout."""
    print("\n" + "=" * 70)
    print("  DATASET VALIDATION REPORT")
    print("=" * 70)
    print("\n── Summary ─────────────────────────────────────────────────────────")
    for k, v in findings["summary"].items():
        print(f"  {k:<32} {v}")

    print("\n── Label distribution ──────────────────────────────────────────────")
    for label, count in findings["label_dist"].items():
        dec = findings["decodable"].get(label, 0)
        print(f"  {label:<15} total={count}  decodable={dec}")

    if findings["errors"]:
        print("\n── ERRORS ───────────────────────────────────────────────────────────")
        for e in findings["errors"]:
            print(f"  [ERROR] {e}")

    if findings["warnings"]:
        print("\n── Warnings ─────────────────────────────────────────────────────────")
        for w in findings["warnings"]:
            print(f"  [WARN]  {w}")

    print("=" * 70 + "\n")
