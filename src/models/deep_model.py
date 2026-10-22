"""
src/models/deep_model.py
-------------------------
CNN-LSTM model for crowd motion classification.

Architecture:
  1. MobileNetV3-Small backbone (pretrained ImageNet, frozen by default)
     extracts a 576-dim feature vector from each sampled frame.
  2. An LSTM receives the ordered sequence of per-frame features and
     produces a final hidden state.
  3. A linear head maps the LSTM output to 2 classes (normal / abnormal).

Design rationale for small dataset:
  - Freezing the CNN means only the LSTM (~100 K params) is trained,
    dramatically reducing overfitting risk with 11 training videos.
  - Pre-trained visual features are far stronger than random init
    on a dataset this tiny.
  - Variable-length sequences are handled by padding + masking.

IMPORTANT: With 11 training videos, this model is expected to overfit.
  Evaluation results on 6 test videos are NOT statistically reliable.
  The model is labelled EXPERIMENTAL in the dashboard.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Label mapping (must match the Random Forest label mapping) ───────────────
LABEL_MAP  = {"normal": 0, "abnormal": 1}
LABEL_NAMES = {0: "normal", 1: "abnormal"}

# ── ImageNet normalisation (required for pretrained MobileNetV3) ─────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Preprocessing for a single BGR frame (numpy uint8 H×W×3) → tensor (3, 224, 224)
FRAME_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class CrowdCNNLSTM(nn.Module):
    """
    MobileNetV3-Small feature extractor + LSTM temporal classifier.

    Parameters
    ----------
    hidden_size  : LSTM hidden units.
    num_layers   : LSTM layers.
    dropout      : Dropout between LSTM layers (ignored for num_layers=1).
    freeze_cnn   : If True, CNN backbone weights are frozen.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        freeze_cnn: bool = True,
    ):
        super().__init__()

        # Load MobileNetV3-Small with pretrained ImageNet weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)
        logger.info("Loaded MobileNetV3-Small (ImageNet pretrained)")

        # Remove the classification head — keep feature extractor only
        # MobileNetV3-Small: features → avgpool → 576-dim vector
        self.cnn = nn.Sequential(
            backbone.features,
            backbone.avgpool,
        )
        self.cnn_output_dim = 576   # fixed for MobileNetV3-Small

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            logger.info("CNN backbone frozen (only LSTM trained)")

        # LSTM temporal classifier
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def extract_frame_features(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from a single preprocessed frame.
        frame_tensor : (3, 224, 224)
        Returns      : (576,)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            out = self.cnn(frame_tensor.unsqueeze(0))   # (1, 576, 1, 1)
            return out.view(-1)                          # (576,)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x       : Padded frame-feature tensor, shape (batch, max_seq, 576).
        lengths : Actual sequence lengths per item in batch, shape (batch,).

        Returns
        -------
        Logits, shape (batch, 2).
        """
        # Pack padded sequence for variable-length handling
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (h_n, _) = self.lstm(packed)   # h_n: (layers, batch, hidden)

        # Take the last layer's hidden state
        h_last = h_n[-1]                          # (batch, hidden)
        out    = self.classifier(self.dropout(h_last))  # (batch, 2)
        return out


def save_deep_model(model: CrowdCNNLSTM, path: Path, extra: dict | None = None) -> None:
    """Save model weights and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "hidden_size":      model.lstm.hidden_size,
        "num_layers":       model.lstm.num_layers,
        "dropout":          model.dropout.p,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Deep model saved to %s", path)


def load_deep_model(path: Path, freeze_cnn: bool = True) -> CrowdCNNLSTM:
    """Load a saved CrowdCNNLSTM checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Deep model checkpoint not found: {path}\n"
            "Run scripts/train_deep_model.py first."
        )
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model = CrowdCNNLSTM(
        hidden_size=payload.get("hidden_size", 128),
        num_layers=payload.get("num_layers", 1),
        dropout=payload.get("dropout", 0.3),
        freeze_cnn=freeze_cnn,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    logger.info("Deep model loaded from %s", path)
    return model
