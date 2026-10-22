"""
scripts/convert_videos.py
--------------------------
Converts non-decodable MOV files to H.264 MP4 using the imageio-ffmpeg
binary installed inside the project venv.

RULES:
  - Originals are NEVER modified or overwritten.
  - Converted copies go to dataset/converted/<folder_name>/
  - Only videos marked decoded=False in the manifest are processed.
  - Each converted video is verified by opening with OpenCV and decoding
    actual frames before being marked as successful.
"""
import sys, subprocess, json, pathlib, cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import imageio_ffmpeg

ROOT = pathlib.Path(__file__).resolve().parent.parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def convert_one(src: pathlib.Path, dst: pathlib.Path) -> dict:
    """Convert a single video to H.264 MP4. Returns status dict."""
    result = {
        "source": str(src),
        "destination": str(dst),
        "conversion_ok": False,
        "decode_ok": False,
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "duration_s": None,
        "error": None,
    }

    # Run FFmpeg conversion
    cmd = [
        FFMPEG, "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",           # drop audio — not needed for motion analysis
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        result["error"] = f"ffmpeg returncode={proc.returncode}: {proc.stderr[-300:]}"
        return result
    result["conversion_ok"] = True

    # Verify the converted file with OpenCV
    if not dst.exists() or dst.stat().st_size == 0:
        result["error"] = "converted file missing or empty"
        return result

    cap = cv2.VideoCapture(str(dst))
    try:
        if not cap.isOpened():
            result["error"] = "opencv cannot open converted file"
            return result
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        if not ret or frame is None:
            result["error"] = "opencv cannot decode first frame of converted file"
            return result
        result["decode_ok"] = True
        result["fps"]         = round(fps, 2) if fps > 0 else None
        result["width"]       = w if w > 0 else None
        result["height"]      = h if h > 0 else None
        result["frame_count"] = fc if fc > 0 else None
        if result["fps"] and result["frame_count"]:
            result["duration_s"] = round(result["frame_count"] / result["fps"], 2)
    finally:
        cap.release()

    return result


def main(test_one: bool = False):
    manifest = ROOT / "dataset/dataset_manifest.csv"
    df = pd.read_csv(manifest)
    bad = df[df["decoded"] == False].copy()

    if bad.empty:
        print("No non-decodable videos found. Nothing to convert.")
        return

    out_root = ROOT / "dataset" / "converted"
    out_root.mkdir(parents=True, exist_ok=True)

    # Test mode: convert only the first labelled normal video
    if test_one:
        labelled_bad = bad[bad["label"].isin(["normal", "abnormal"])]
        if labelled_bad.empty:
            print("No labelled non-decodable videos to test with.")
            return
        row = labelled_bad.iloc[0]
        src = pathlib.Path(row["path"])
        folder = row["folder"].replace(" ", "_")
        dst = out_root / folder / (src.stem + "_converted.mp4")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"TEST CONVERSION: {src.name} -> {dst.name}")
        result = convert_one(src, dst)
        print(json.dumps(result, indent=2))
        if result["decode_ok"]:
            print("\nTEST PASSED — converted video decodes correctly.")
        else:
            print(f"\nTEST FAILED — {result['error']}")
        return

    # Full conversion: all 9 files
    print(f"Converting {len(bad)} non-decodable videos...")
    print("=" * 80)
    results = []
    for _, row in bad.iterrows():
        src = pathlib.Path(row["path"])
        if not src.exists():
            print(f"  SKIP (file missing): {row['filename']}")
            continue
        folder = row["folder"].replace(" ", "_")
        dst = out_root / folder / (src.stem + "_converted.mp4")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            print(f"  SKIP (already converted): {dst.name}")
            # Still verify it
            result = convert_one(src, dst)  # re-convert to verify
            # Actually just verify the existing file
            cap = cv2.VideoCapture(str(dst))
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                print(f"    Already converted and decodable — skipping")
                results.append({
                    "source": row["filename"],
                    "label": row["label"],
                    "converted": dst.name,
                    "conversion_ok": True,
                    "decode_ok": True,
                })
                continue

        print(f"  Converting: {src.name} ...", end=" ", flush=True)
        result = convert_one(src, dst)
        result["source_filename"] = row["filename"]
        result["label"] = row["label"]
        results.append(result)

        if result["decode_ok"]:
            print(f"OK  {result['width']}x{result['height']}  {result['fps']}fps  {result['frame_count']} frames  {result['duration_s']}s")
        else:
            print(f"FAILED  {result.get('error','unknown')}")

    # Summary
    print()
    print("=" * 80)
    ok_count = sum(1 for r in results if r.get("decode_ok"))
    fail_count = sum(1 for r in results if not r.get("decode_ok"))
    print(f"Conversion complete: {ok_count} succeeded, {fail_count} failed")

    # By label
    for label in ["normal", "abnormal", "unknown"]:
        label_results = [r for r in results if r.get("label") == label]
        label_ok = sum(1 for r in label_results if r.get("decode_ok"))
        print(f"  {label}: {label_ok}/{len(label_results)} recovered")

    # Save report
    report_path = ROOT / "outputs" / "reports" / "conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-one", action="store_true",
                        help="Test conversion on one labelled file only")
    args = parser.parse_args()
    main(test_one=args.test_one)
