"""
scripts/train_deep_model.py
----------------------------
Trains the CNN-LSTM crowd anomaly classifier.

Design choices:
  - MobileNetV3-Small backbone frozen (only LSTM + classifier trained).
  - Validation-based best-checkpoint selection.
  - Early stopping (patience=10).
  - Weighted cross-entropy to handle class imbalance.
  - Deterministic seed for reproducibility.
  - Per-epoch train/val loss and accuracy logged.

Run:
    .\.venv\Scripts\python.exe scripts/train_deep_model.py

Produces:
    models/deep_model.pth
    outputs/reports/deep_train_history.json
"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.utils.paths import load_config, get_path, PROJECT_ROOT
from src.utils.logging_config import setup_logging, get_logger
from src.data.inventory import load_inventory
from src.models.deep_model import CrowdCNNLSTM, save_deep_model
from src.models.deep_dataset import VideoFrameDataset, collate_variable_length

cfg = load_config()
setup_logging(cfg["logging"]["level"])
logger = get_logger("train_deep_model")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = cfg["training"].get("random_seed", 42)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS       = 40
LR           = 3e-4
HIDDEN_SIZE  = 128
NUM_LAYERS   = 1
DROPOUT      = 0.3
FREEZE_CNN   = True
PATIENCE     = 10      # early stopping patience
SAMPLE_INT   = cfg["preprocessing"]["frame_sample_interval"]
MAX_FRAMES   = 150
DEVICE       = torch.device("cpu")


def main():
    logger.info("=" * 60)
    logger.info("CNN-LSTM Training")
    logger.info("=" * 60)

    # ── Load manifest ─────────────────────────────────────────────────────
    manifest_path = PROJECT_ROOT / cfg["dataset"]["manifest_path"]
    df = load_inventory(manifest_path)

    # ── Build model ───────────────────────────────────────────────────────
    logger.info("Building CNN-LSTM model...")
    model = CrowdCNNLSTM(
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=DROPOUT, freeze_cnn=FREEZE_CNN,
    ).to(DEVICE)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d total", trainable_params, total_params)

    # ── Build datasets ────────────────────────────────────────────────────
    train_ds = VideoFrameDataset(df, model, split="train",
                                 sample_interval=SAMPLE_INT, max_frames=MAX_FRAMES)
    val_ds   = VideoFrameDataset(df, model, split="val",
                                 sample_interval=SAMPLE_INT, max_frames=MAX_FRAMES)

    if len(train_ds) == 0:
        logger.error("Training set is empty. Run prepare_dataset.py first.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_variable_length)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_variable_length)

    # Class weights to handle imbalance
    from src.models.deep_model import LABEL_MAP
    labels_train = [LABEL_MAP[row["label"]]
                    for _, row in train_ds.records.iterrows()
                    if row["label"] in LABEL_MAP]
    from collections import Counter
    counts = Counter(labels_train)
    total  = sum(counts.values())
    weights = torch.tensor([total / (2 * counts.get(i, 1)) for i in range(2)],
                           dtype=torch.float)
    logger.info("Class weights: %s", weights.tolist())

    from src.models.deep_model import LABEL_MAP
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": [],
                     "train_acc": [], "val_acc": [], "epochs": []}
    ckpt_path = get_path(cfg, "training", "model_output_dir") / "deep_model.pth"

    logger.info("Training for up to %d epochs (patience=%d)...", EPOCHS, PATIENCE)
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for padded, lengths, labels_t, _ in train_loader:
            padded   = padded.to(DEVICE)
            lengths  = lengths.to(DEVICE)
            labels_t = labels_t.to(DEVICE)

            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss   = criterion(logits, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss    += loss.item()
            preds       = logits.argmax(dim=1)
            tr_correct += (preds == labels_t).sum().item()
            tr_total   += len(labels_t)

        tr_acc   = tr_correct / tr_total if tr_total else 0.0
        tr_loss  /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for padded, lengths, labels_t, _ in val_loader:
                padded   = padded.to(DEVICE)
                lengths  = lengths.to(DEVICE)
                labels_t = labels_t.to(DEVICE)
                logits   = model(padded, lengths)
                val_loss += criterion(logits, labels_t).item()
                preds     = logits.argmax(dim=1)
                val_correct += (preds == labels_t).sum().item()
                val_total   += len(labels_t)

        val_acc  = val_correct / val_total if val_total else 0.0
        val_loss /= max(len(val_loader), 1)

        history["epochs"].append(epoch)
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            "Epoch %3d/%d  tr_loss=%.4f  tr_acc=%.3f  val_loss=%.4f  val_acc=%.3f",
            epoch, EPOCHS, tr_loss, tr_acc, val_loss, val_acc
        )

        # Checkpoint on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            save_deep_model(model, ckpt_path, extra={
                "epoch": epoch, "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "seed": SEED, "sample_interval": SAMPLE_INT,
                "max_frames": MAX_FRAMES,
            })
            logger.info("  ✓ Checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    elapsed = time.perf_counter() - t_start
    logger.info("Training finished in %.1f s. Best val_loss=%.4f", elapsed, best_val_loss)
    logger.info("Checkpoint: %s", ckpt_path)

    # Save history
    history_path = PROJECT_ROOT / "outputs" / "reports" / "deep_train_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    print(f"\n  CNN-LSTM training complete.")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Next: python scripts/evaluate_deep_model.py")


if __name__ == "__main__":
    main()
