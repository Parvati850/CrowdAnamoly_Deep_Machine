"""
src/inference/realtime.py
--------------------------
Real-time inference from a webcam or RTSP stream.
Uses a rolling buffer of sampled frames and the same feature pipeline
as offline inference.

Thread-safety: VideoStream runs capture in a background thread.
               The main loop reads the latest frame from a shared queue.
"""

import cv2
import time
import threading
import queue
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from src.inference.video_inference import _features_from_gray_buffer
from src.models.baseline import load_model, predict as model_predict
from src.data.extract_frames import resize_frame
from src.inference.alerts import AlertManager
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, get_path

logger = get_logger(__name__)

LABEL_NAMES = {0: "normal", 1: "abnormal"}


class VideoStream:
    """
    Background-threaded video capture.
    Reads frames from a camera or RTSP source and puts them in a queue.
    The queue is bounded to prevent unbounded memory growth.
    """

    def __init__(self, source, queue_size: int = 64):
        self.source     = source
        self._queue     = queue.Queue(maxsize=queue_size)
        self._stop_flag = threading.Event()
        self._thread    = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> "VideoStream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_flag.set()
        self._thread.join(timeout=3.0)

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self.source)
            return

        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Stream ended or capture failed.")
                break
            # Discard oldest if queue full (keep newest)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

        cap.release()


class RealtimeAnalyser:
    """
    Runs the inference loop over a live video stream.

    Parameters
    ----------
    cfg             : Config dict.
    model           : Pre-loaded sklearn pipeline.
    alert_manager   : AlertManager instance.
    on_result       : Callback fn(score, label, fps) called after each window.
    window_size     : Number of sampled frames per analysis window.
    """

    def __init__(
        self,
        cfg: dict,
        model,
        alert_manager: AlertManager,
        on_result: Optional[Callable] = None,
        window_size: int = 20,
    ):
        self.cfg           = cfg
        self.model         = model
        self.alert_manager = alert_manager
        self.on_result     = on_result
        self.window_size   = window_size

        self._interval = cfg["preprocessing"]["frame_sample_interval"]
        self._width    = cfg["preprocessing"]["target_width"]
        self._height   = cfg["preprocessing"]["target_height"]
        self._running  = False
        self._stream: Optional[VideoStream] = None

    def start(self, source) -> None:
        """
        Begin processing frames from `source` (webcam index or RTSP URL).
        Blocks until stop() is called or the stream ends.
        """
        self._running = True
        self._stream  = VideoStream(source, self.cfg["inference"]["queue_size"])
        self._stream.start()
        self._run_loop(source)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()

    def _run_loop(self, source) -> None:
        buffer    = []
        frame_idx = 0
        t_fps     = time.perf_counter()
        fps_count = 0

        while self._running:
            frame = self._stream.read(timeout=0.05)
            if frame is None:
                continue

            if frame_idx % self._interval == 0:
                resized = resize_frame(frame, self._width, self._height)
                gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                buffer.append(gray)
                fps_count += 1

                if len(buffer) >= self.window_size:
                    feat = _features_from_gray_buffer(buffer)
                    if feat is not None:
                        X = feat.reshape(1, -1)
                        try:
                            preds, probs = model_predict(self.model, X)
                            score = float(probs[0, 1])
                            label = LABEL_NAMES.get(int(preds[0]), "unknown")
                        except Exception as exc:
                            logger.warning("Realtime prediction error: %s", exc)
                            score, label = 0.0, "unknown"

                        # source_name is already set on the AlertManager at construction;
                        # no need to pass it per-call
                        self.alert_manager.update(score, label)

                        elapsed = time.perf_counter() - t_fps
                        fps = fps_count / elapsed if elapsed > 0 else 0.0
                        fps_count = 0
                        t_fps     = time.perf_counter()

                        if self.on_result:
                            self.on_result(score, label, round(fps, 1))

                    buffer = []

            frame_idx += 1

        if self._stream:
            self._stream.stop()
