"""
scripts/inspect_dataset.py
---------------------------
Phase 1: Dataset inspection.

Run:
    python scripts/inspect_dataset.py

Produces:
    dataset/dataset_manifest.csv
    docs/DATASET_REPORT.md
    outputs/reports/validation_findings.json
"""

import sys
import json
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, get_path, PROJECT_ROOT, ensure_dir
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import build_inventory, save_inventory
from src.data.validate import validate_manifest, print_validation_report

def main():
    cfg = load_config()
    setup_logging(cfg["logging"]["level"],
                  PROJECT_ROOT / cfg["logging"]["log_file"])
    logger = get_logger("inspect_dataset")

    logger.info("=" * 60)
    logger.info("Phase 1: Dataset Inspection")
    logger.info("=" * 60)

    # ── Build inventory ──────────────────────────────────────────────────
    df = build_inventory(cfg)

    if df.empty:
        logger.error("No videos found. Verify your dataset path in config/config.yaml.")
        sys.exit(1)

    # ── Save manifest ────────────────────────────────────────────────────
    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    save_inventory(df, manifest_path)

    # ── Validate ─────────────────────────────────────────────────────────
    findings = validate_manifest(df)
    print_validation_report(findings)

    # Save validation findings
    reports_dir = get_path(cfg, "output", "reports_dir")
    ensure_dir(reports_dir)
    findings_path = reports_dir / "validation_findings.json"
    with open(findings_path, "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, default=str)
    logger.info("Validation findings saved to %s", findings_path)

    # ── Print full inventory table ────────────────────────────────────────
    print("\n── Full Inventory ───────────────────────────────────────────────────")
    cols = ["filename", "folder", "label", "decoded", "fps", "duration_s", "size_mb"]
    print(df[cols].to_string(index=False))

    # ── Write DATASET_REPORT.md ────────────────────────────────────────────
    docs_dir = PROJECT_ROOT / "docs"
    ensure_dir(docs_dir)
    _write_dataset_report(df, findings, docs_dir / "DATASET_REPORT.md")
    logger.info("DATASET_REPORT.md written.")

    logger.info("Phase 1 complete. Manifest saved to %s", manifest_path)
    return df, findings


def _write_dataset_report(df, findings, output_path):
    s = findings["summary"]
    lines = [
        "# Dataset Report",
        "",
        f"Generated automatically by `scripts/inspect_dataset.py`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total videos discovered | {s['total_videos']} |",
        f"| Labelled videos | {s['labelled_videos']} |",
        f"| Extra (unlabelled) videos | {s['extra_videos']} |",
        f"| Normal videos | {s['normal_videos']} |",
        f"| Abnormal videos | {s['abnormal_videos']} |",
        f"| Fully decodable | {s['fully_decodable']} |",
        f"| Non-decodable | {s['non_decodable']} |",
        "",
        "## Per-class decodable videos",
        "",
        "| Label | Total | Decodable |",
        "|-------|-------|-----------|",
    ]
    for label, count in findings["label_dist"].items():
        dec = findings["decodable"].get(label, 0)
        lines.append(f"| {label} | {count} | {dec} |")

    lines += [
        "",
        "## Warnings",
        "",
    ]
    for w in findings["warnings"]:
        lines.append(f"- {w}")

    if findings["errors"]:
        lines += ["", "## Errors", ""]
        for e in findings["errors"]:
            lines.append(f"- **{e}**")

    lines += [
        "",
        "## Video Inventory",
        "",
        "| Filename | Folder | Label | Decoded | FPS | Duration (s) | Size (MB) |",
        "|----------|--------|-------|---------|-----|-------------|-----------|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['filename']} | {row['folder']} | {row['label']} "
            f"| {row['decoded']} | {row.get('fps','?')} "
            f"| {row.get('duration_s','?')} | {row.get('size_mb','?')} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- The labelled dataset contains very few videos (< 25 labelled). "
          "Supervised learning results should be interpreted with caution.",
        "- The 'Crowd Dataset - extra' folder has no provided labels. "
          "These videos were not used for training.",
        "- Some .mov files use a zlib-compressed container format that "
          "OpenCV's built-in MSMF backend cannot decode. "
          "Installing FFmpeg or opencv-python-headless may help.",
        "- Short video clips (< 5 s) provide very few frame pairs for "
          "motion feature extraction.",
    ]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
