# Dataset Report

Generated automatically by `scripts/inspect_dataset.py`

## Summary

| Metric | Value |
|--------|-------|
| Total videos discovered | 55 |
| Labelled videos | 23 |
| Extra (unlabelled) videos | 23 |
| Normal videos | 15 |
| Abnormal videos | 8 |
| Fully decodable | 46 |
| Non-decodable | 9 |

## Per-class decodable videos

| Label | Total | Decodable |
|-------|-------|-----------|
| unknown | 32 | 26 |
| normal | 15 | 12 |
| abnormal | 8 | 8 |

## Warnings

- Cannot decode: 81872953_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: sfw20110024_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: shibuya1.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: 1379-7_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: 1379-8_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: 81712424_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: 82409835_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: 82409905_l.mov | Error: first_frame_decode_failed (codec not supported)
- Cannot decode: 965-83_l.mov | Error: first_frame_decode_failed (codec not supported)
- Duplicate filenames across folders: ['81872953_l_converted.mp4', 'sfw20110024_l_converted.mp4', 'shibuya1_converted.mp4', '81872953_l_converted.mp4', 'sfw20110024_l_converted.mp4', 'shibuya1_converted.mp4']

## Video Inventory

| Filename | Folder | Label | Decoded | FPS | Duration (s) | Size (MB) |
|----------|--------|-------|---------|-----|-------------|-----------|
| 1183-88_l.mov | Abnormal Crowds | abnormal | True | 25.0 | 20.6 | 5.714 |
| 2010-291_l.avi | Abnormal Crowds | abnormal | True | 25.0 | 4.32 | 53.396 |
| 2014-140_l.avi | Abnormal Crowds | abnormal | True | 25.0 | 6.88 | 85.038 |
| 2017-420_l.mov | Abnormal Crowds | abnormal | True | 25.004 | 10.2 | 4.381 |
| 263C044_060_c.mov | Abnormal Crowds | abnormal | True | 23.976 | 11.3 | 24.284 |
| 263C044_064_c.mov | Abnormal Crowds | abnormal | True | 23.976 | 20.15 | 50.384 |
| 3452204_031_c.mov | Abnormal Crowds | abnormal | True | 23.976 | 28.15 | 61.335 |
| 490-208_l.mov | Abnormal Crowds | abnormal | True | 25.016 | 2.64 | 0.736 |
| 341-46_l.mov | Normal Crowds | normal | True | 25.004 | 9.96 | 0.844 |
| 3687-18_70.mov | Normal Crowds | normal | True | 25.0 | 18.56 | 4.669 |
| 637-147_l.mov | Normal Crowds | normal | True | 25.009 | 4.52 | 0.875 |
| 688-10_l.mov | Normal Crowds | normal | True | 25.007 | 5.76 | 2.421 |
| 81872953_l.mov | Normal Crowds | normal | False | 24.0 | nan | 6.216 |
| 81872953_l_converted.mp4 | Normal Crowds | normal | True | 24.0 | 35.08 | 5.166 |
| 821-10_l.mov | Normal Crowds | normal | True | 25.0 | 50.12 | 5.812 |
| 879-38_l.mov | Normal Crowds | normal | True | 25.0 | 51.4 | 14.191 |
| 9019-13_l.mov | Normal Crowds | normal | True | 24.0 | 22.25 | 0.897 |
| 906-17_l.mov | Normal Crowds | normal | True | 25.0 | 15.76 | 4.866 |
| CRW116.mov | Normal Crowds | normal | True | 29.97 | 21.02 | 1.735 |
| sfw20110024_l.mov | Normal Crowds | normal | False | 30.038 | nan | 1.741 |
| sfw20110024_l_converted.mp4 | Normal Crowds | normal | True | 30.0 | 14.6 | 1.196 |
| shibuya1.mov | Normal Crowds | normal | False | 30.0 | nan | 21.744 |
| shibuya1_converted.mp4 | Normal Crowds | normal | True | 29.98 | 102.43 | 15.598 |
| 1379-7_l_converted.mp4 | Crowd_Dataset_-_extra | unknown | True | 24.0 | 10.21 | 1.432 |
| 1379-8_l_converted.mp4 | Crowd_Dataset_-_extra | unknown | True | 23.935 | 30.96 | 3.461 |
| 81712424_l_converted.mp4 | Crowd_Dataset_-_extra | unknown | True | 30.0 | 6.13 | 0.968 |
| 82409835_l_converted.mp4 | Crowd_Dataset_-_extra | unknown | True | 24.0 | 16.42 | 2.491 |
| 82409905_l_converted.mp4 | Crowd_Dataset_-_extra | unknown | True | 24.0 | 30.96 | 6.761 |
| 965-83_l_converted.mp4 | Crowd_Dataset_-_extra | unknown | True | 24.0 | 6.17 | 1.403 |
| 81872953_l_converted.mp4 | Normal_Crowds | unknown | True | 24.0 | 35.08 | 5.166 |
| sfw20110024_l_converted.mp4 | Normal_Crowds | unknown | True | 30.0 | 14.6 | 1.196 |
| shibuya1_converted.mp4 | Normal_Crowds | unknown | True | 29.98 | 102.43 | 15.598 |
| 1379-7_l.mov | Crowd Dataset - extra | unknown | False | 24.0 | nan | 1.474 |
| 1379-8_l.mov | Crowd Dataset - extra | unknown | False | 24.0 | nan | 3.304 |
| 4565-6.wmv | Crowd Dataset - extra | unknown | True | 25.0 | 27.0 | 6.498 |
| 663-13_l.mov | Crowd Dataset - extra | unknown | True | 25.009 | 4.88 | 0.79 |
| 664-55_l-1.mov | Crowd Dataset - extra | unknown | True | 25.007 | 5.56 | 0.989 |
| 7399-1_70.mov | Crowd Dataset - extra | unknown | True | 25.0 | 15.44 | 3.764 |
| 7808-28.wmv | Crowd Dataset - extra | unknown | True | 25.0 | 8.12 | 1.89 |
| 81712424_l.mov | Crowd Dataset - extra | unknown | False | 30.0 | nan | 1.445 |
| 82409835_l.mov | Crowd Dataset - extra | unknown | False | 24.0 | nan | 3.226 |
| 82409905_l.mov | Crowd Dataset - extra | unknown | False | 24.0 | nan | 10.915 |
| 861-13_70.mov | Crowd Dataset - extra | unknown | True | 25.0 | 20.12 | 4.859 |
| 965-83_l.mov | Crowd Dataset - extra | unknown | False | 24.0 | nan | 2.051 |
| CRW104.mov | Crowd Dataset - extra | unknown | True | 29.97 | 17.08 | 1.427 |
| CRW131.mov | Crowd Dataset - extra | unknown | True | 29.97 | 11.01 | 0.872 |
| CRW132.mov | Crowd Dataset - extra | unknown | True | 29.97 | 10.01 | 0.863 |
| data1.avi | Crowd Dataset - extra | unknown | True | 15.0 | 3.67 | 0.876 |
| data2.mov | Crowd Dataset - extra | unknown | True | 25.0 | 4.64 | 0.907 |
| data6.mov | Crowd Dataset - extra | unknown | True | 25.007 | 6.28 | 1.204 |
| data7.mov | Crowd Dataset - extra | unknown | True | 25.013 | 3.32 | 0.63 |
| flash3.mov | Crowd Dataset - extra | unknown | True | 30.0 | 78.7 | 61.537 |
| flash4.mov | Crowd Dataset - extra | unknown | True | 15.0 | 24.0 | 12.343 |
| LiveLeak-dot-com-57306-police.flv | Crowd Dataset - extra | unknown | True | 15.0 | 30.87 | 1.191 |
| UCF_tailgating_part_NGA_1.00.avi | Crowd Dataset - extra | unknown | True | 29.97 | 142.34 | 35.814 |

## Limitations

- The labelled dataset contains very few videos (< 25 labelled). Supervised learning results should be interpreted with caution.
- The 'Crowd Dataset - extra' folder has no provided labels. These videos were not used for training.
- Some .mov files use a zlib-compressed container format that OpenCV's built-in MSMF backend cannot decode. Installing FFmpeg or opencv-python-headless may help.
- Short video clips (< 5 s) provide very few frame pairs for motion feature extraction.