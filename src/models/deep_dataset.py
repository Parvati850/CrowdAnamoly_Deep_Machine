"""
src/models/deep_dataset.py
---------------------------
PyTorch Dataset and collation for the CNN-LSTM pipeline.

Each sample is one video:
  - Frames are sampled at a configurable interval.
  - Each frame is preprocessed with the ImageNet transform.
  - The sequence of CNN feature vectors is the model input.
  - Label: 0=normal, 1=abnormal.

Variable-length sequences are returned as padded tensors by the
custom collate function, along with the actual lengths for pack_padded_sequence.
"""

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional

from src.models.deep_model import FRAME_TRANSFORM, LABEL_MAP, CrowdCNNLSTM
from src.data.extract_frames import resize_frame
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class VideoFrameDataset(Dataset):
    """
    Loads frames from labelled videos, extracts CNN features on-the-fly.

    Parameters
    ----------
    manifest        : DataFrame with columns path, label, decoded, split.
    split           : 'train' | 'val' | 'test' | None (all labelled decodable).
    cnn_model       : Frozen CrowdCNNLSTM (used for frame feature extraction).
    sample_interval : Extract every Nth frame.
    max_frames      : Cap sequence length (avoids OOM on very long videos).
    device          : torch device.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cnn_model: CrowdCNNLSTM,
        split: Optional[str] = None,
        sample_interval: int = 10,
        max_frames: int = 150,
        device: torch.device = torch.device("cpu"),
    ):
        self.cnn            = cnn_model
        self.sample_interval = sample_interval
        self.max_frames     = max_frames
        self.device         = device

        subset = manifest[
            (manifest["source_group"] == "labelled") &
            (manifest["decoded"] == True)
        ]
        if split is not None:
            subset = subset[subset["split"] == split]

        self.records = subset.reset_index(drop=True)
        logger.info(
            "VideoFrameDataset: split=%s  videos=%d",
            split or "all", len(self.records)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Returns
        -------
        features : (seq_len, 576) float tensor of CNN frame features
        label    : int (0=normal, 1=abnormal)
        filename : str
        """
        row       = self.records.iloc[idx]
        video_path = Path(row["path"])
        label      = LABEL_MAP[row["label"]]
        filename   = row["filename"]

        features = _extract_frame_features(
            video_path, self.cnn, self.sample_interval,
            self.max_frames, self.device
        )

        if features is None:
            # Return a zero sequence of length 1 — model will handle gracefully
            features = torch.zeros(1, self.cnn.cnn_output_dim)
            logger.warning("Feature extraction failed for %s — using zeros", filename)

        return features, label, filename


def _extract_frame_features(
    video_path: Path,
    cnn: CrowdCNNLSTM,
    interval: int,
    max_frames: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Read video, sample frames, run through frozen CNN, return feature sequence.
    Returns None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    feature_list = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                # Convert BGR→RGB, apply ImageNet transform
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = FRAME_TRANSFORM(frame_rgb).to(device)  # (3, 224, 224)
                with torch.no_grad():
                    feat = cnn.extract_frame_features(tensor)    # (576,)
                feature_list.append(feat.cpu())
                if len(feature_list) >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    if len(feature_list) < 2:
        return None

    return torch.stack(feature_list, dim=0).float()   # (seq_len, 576)


def collate_variable_length(
    batch: list[tuple[torch.Tensor, int, str]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Collate variable-length frame sequences into a padded batch.

    Returns
    -------
    padded_features : (batch, max_seq, 576)
    lengths         : (batch,) actual sequence lengths
    labels          : (batch,) integer labels
    filenames       : list of str
    """
    features, labels, filenames = zip(*batch)
    lengths  = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    padded   = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels_t, list(filenames)
