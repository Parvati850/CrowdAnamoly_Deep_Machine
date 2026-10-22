"""
src/utils/logging_config.py
----------------------------
Configures the root logger once.  All modules should call get_logger(__name__)
instead of creating their own handlers.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure root logger with a console handler and an optional file handler.

    Parameters
    ----------
    level    : Logging level string: DEBUG, INFO, WARNING, ERROR
    log_file : Optional path to write log output to a file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call setup_logging() first at application entry."""
    return logging.getLogger(name)
