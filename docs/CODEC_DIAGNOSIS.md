# Codec Diagnosis Report

## Summary

9 of 43 video files cannot be decoded by OpenCV on Windows.
The failure is **not** a corrupt file, incorrect path, or wrong extension.
It is a known codec/container compatibility limitation.

---

## Exact Error Messages

OpenCV MSMF backend prints:
```
[mov,mp4,m4a,3gp,3g2,mj2 @ ...] this file requires zlib support compiled in
[mov,mp4,m4a,3gp,3g2,mj2 @ ...] error reading header
```

The `cap.isOpened()` call returns `True` (the file can be opened),
but `cap.read()` returns `(False, None)` — no frame can be decoded.

The project records this as `decode_error = "first_frame_decode_failed (codec not supported)"`.

---

## Root Cause

These `.mov` files use a **zlib-compressed QuickTime container** (also called
"fast-start" or "compressed movie header" format). The container metadata
itself is compressed with zlib.

OpenCV on Windows uses the **Microsoft Media Foundation (MSMF)** backend
by default. The bundled FFmpeg in `opencv-python` on Windows is compiled
**without zlib container support** for this specific QuickTime variant.

This is confirmed by the FFmpeg error `this file requires zlib support compiled in`.

---

## Affected Files

| Filename | Folder | Label |
|---|---|---|
| `81872953_l.mov` | Normal Crowds | normal |
| `sfw20110024_l.mov` | Normal Crowds | normal |
| `shibuya1.mov` | Normal Crowds | normal |
| `1379-7_l.mov` | Crowd Dataset - extra | unknown |
| `1379-8_l.mov` | Crowd Dataset - extra | unknown |
| `81712424_l.mov` | Crowd Dataset - extra | unknown |
| `82409835_l.mov` | Crowd Dataset - extra | unknown |
| `82409905_l.mov` | Crowd Dataset - extra | unknown |
| `965-83_l.mov` | Crowd Dataset - extra | unknown |

Impact on training: **3 normal labelled videos excluded** (9 decodable normal remain).
All 8 abnormal labelled videos are decodable.

---

## Why Renaming the Extension Does Not Help

Changing `.mov` to `.mp4` does not re-encode the file.
The container format and compression remain identical.
OpenCV will still fail to decode the header.

---

## How to Fix (Optional Conversion Utility)

If you want to use these files, convert them with a **zlib-enabled FFmpeg**
binary (e.g., from https://www.gyan.dev/ffmpeg/builds/ — the "full" build
includes zlib support).

A conversion utility is provided at:

```
scripts/convert_unreadable_videos.py
```

It writes converted copies to `generated_data/converted/` and **never
overwrites the originals**. Run it only if you want to attempt recovery.

---

## Verification

All 9 files were tested with `cv2.VideoCapture` + `cap.read()`.
The same test confirms all other 34 videos decode successfully.
The dataset manifest (`dataset/dataset_manifest.csv`) records the
`decoded` and `decode_error` columns for every file.

---

## Impact on Project

- Training uses only the 17 labelled decodable videos.
- The 9 non-decodable files are excluded automatically.
- The manifest and all scripts handle this gracefully.
- Non-decodable files are never silently skipped — they appear in the manifest
  with `decoded=False` and a specific `decode_error` reason.
