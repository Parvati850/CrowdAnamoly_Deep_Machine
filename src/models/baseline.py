"""
src/models/baseline.py
-----------------------
Baseline supervised classifier using motion features + scikit-learn.

WHY A BASELINE FIRST?
  With only ~20 labelled videos, deep learning risks severe overfitting.
  A Random Forest or SVM on hand-crafted motion features is interpretable,
  fast to train, and gives a reliable benchmark.

The pipeline includes:
  - StandardScaler (fit only on training data)
  - Classifier (configurable via config.yaml)

A full sklearn Pipeline is used so that the scaler is automatically applied
at inference without any manual step.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _build_svm(cfg: dict):
    """
    Build a probability-calibrated SVM.
    sklearn >= 1.9 deprecates SVC(probability=True).
    CalibratedClassifierCV wraps a plain SVC and adds probability outputs.
    cv=2 allows training on small datasets (as few as 2 samples per class).
    """
    from sklearn.svm import SVC as _SVC
    base = _SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=cfg["training"].get("random_seed", 42),
    )
    # cv=2 ensures the calibration cross-validation works even on tiny datasets
    return CalibratedClassifierCV(base, ensemble=False, cv=2)


CLASSIFIERS = {
    "random_forest": lambda cfg: RandomForestClassifier(
        n_estimators=cfg["model"].get("n_estimators", 100),
        max_depth=cfg["model"].get("max_depth") or None,
        class_weight="balanced",
        random_state=cfg["training"].get("random_seed", 42),
    ),
    "svm": lambda cfg: _build_svm(cfg),
    "logistic": lambda cfg: LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=cfg["training"].get("random_seed", 42),
    ),
}


def build_pipeline(cfg: dict) -> Pipeline:
    """
    Construct a sklearn Pipeline: StandardScaler → Classifier.

    The classifier type is set by cfg['model']['baseline_classifier'].
    Defaults to 'random_forest'.
    """
    clf_name = cfg["model"].get("baseline_classifier", "random_forest")
    if clf_name not in CLASSIFIERS:
        logger.warning(
            "Unknown classifier '%s'. Falling back to random_forest.", clf_name
        )
        clf_name = "random_forest"

    clf = CLASSIFIERS[clf_name](cfg)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    logger.info("Built pipeline: StandardScaler → %s", type(clf).__name__)
    return pipeline


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> Pipeline:
    """
    Fit the baseline pipeline on training data.

    Parameters
    ----------
    X_train : Feature matrix, shape (N, n_features)
    y_train : Integer labels (0=normal, 1=abnormal)
    cfg     : Config dict.

    Returns
    -------
    Fitted Pipeline.
    """
    if len(X_train) == 0:
        raise ValueError("Training set is empty. Cannot train the model.")

    pipeline = build_pipeline(cfg)
    logger.info(
        "Training baseline on %d samples (%d features)...",
        len(X_train), X_train.shape[1]
    )
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")
    return pipeline


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Serialize a fitted pipeline to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> Pipeline:
    """Load a serialized pipeline from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}\n"
            "Run scripts/train_model.py first."
        )
    with open(path, "rb") as fh:
        pipeline = pickle.load(fh)
    logger.info("Model loaded from %s", path)
    return pipeline


def predict(pipeline: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on feature matrix X.

    Returns
    -------
    (labels, probabilities)
      labels : integer predictions (0=normal, 1=abnormal)
      probs  : float array of shape (N, 2), columns [P(normal), P(abnormal)]
    """
    labels = pipeline.predict(X)
    probs  = pipeline.predict_proba(X)
    return labels, probs
