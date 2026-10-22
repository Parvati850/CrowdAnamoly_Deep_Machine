"""
scripts/convert_unreadable_videos.py
--------------------------------------
OPTIONAL utility: converts non-decodable videos to H.264 MP4 using
an external FFmpeg binary.

RULES:
  - Originals are NEVER modified or overwritten.
  - Converted copies go to generated_data/converted/<original_folder>/.
  - Only videos marked decoded=False in the manifest are processed.
  - Requires FFmpeg to be installed and on PATH (not the one bundled with OpenCV).

Usage:
    python scripts/convert_unreadable_videos.py

After conversion, re-run inspect_dataset.py to update the manifest.
"""

import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import load_config, PROJECT_ROOT
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import load_inventory


def main():
    cfg = load_config()
    setup_logging("INFO")
    logger = get_logger("convert_unreadable_videos")

    # Check FFmpeg is available
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.error(
            "FFmpeg not found on PATH.\n"
            "Download a full build from https://www.gyan.dev/ffmpeg/builds/\n"
            "and add the 'bin' folder to your system PATH."
        )
        sys.exit(1)
    logger.info("Using FFmpeg: %s", ffmpeg)

    # Load manifest
    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    df = load_inventory(manifest_path)

    non_decodable = df[df["decoded"] == False].copy()
    if non_decodable.empty:
        logger.info("No non-decodable videos found in manifest. Nothing to convert.")
        return

    out_root = PROJECT_ROOT / "generated_data" / "converted"
    out_root.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    for _, row in non_decodable.iterrows():
        src = Path(row["path"])
        if not src.exists():
            logger.warning("Source not found, skipping: %s", src)
            continue

        folder = row["folder"].replace(" ", "_")
        out_dir = out_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (src.stem + "_converted.mp4")

        if dst.exists():
            logger.info("Already converted, skipping: %s", dst.name)
            success += 1
            continue

        logger.info("Converting: %s → %s", src.name, dst.name)
        cmd = [
            ffmpeg, "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",   # drop audio — we only need video frames
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("  ✓ Converted: %s", dst.name)
            success += 1
        else:
            logger.warning("  ✗ Failed: %s\n    %s", src.name,
                           result.stderr[-300:] if result.stderr else "(no output)")
            failed += 1

    logger.info(
        "Conversion complete. Success=%d  Failed=%d  Output dir: %s",
        success, failed, out_root,
    )
    if success:
        logger.info(
            "Run 'python scripts/inspect_dataset.py' with the converted folder "
            "added to config to include these videos in the manifest."
        )


if __name__ == "__main__":
    main()
