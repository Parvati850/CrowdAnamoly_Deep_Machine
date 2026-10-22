"""
src/utils/paths.py
------------------
Single source of truth for all resolved filesystem paths.
Import PROJECT_ROOT and helper functions from here; never hardcode paths
in other modules.
"""

from pathlib import Path
import yaml

# ── Project root is the directory containing this utils package's parent ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # .../CrowdAnomalyDetection


def load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml and return as a plain dict."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please ensure config/config.yaml exists in the project root."
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_dataset_root(cfg: dict) -> Path:
    p = Path(cfg["dataset"]["dataset_root"])
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {p}\n"
            "Please update 'dataset.dataset_root' in config/config.yaml."
        )
    return p


def get_path(cfg: dict, *keys: str) -> Path:
    """
    Resolve a config value to an absolute Path.
    Keys are dot-separated config hierarchy, e.g. get_path(cfg, 'output', 'plots_dir').
    Values that are relative are resolved from PROJECT_ROOT.
    """
    node = cfg
    for k in keys:
        node = node[k]
    p = Path(node)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
