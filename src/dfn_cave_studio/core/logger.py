"""Logging management for DFN Cave Studio."""

import logging
import sys
from pathlib import Path
from typing import Optional

from dfn_cave_studio.core.config import get_config


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return the root logger for DFN Cave Studio.

    Args:
        log_file: Path to log file. If None, uses the path from config.

    Returns:
        Configured root logger instance.
    """
    config = get_config()
    logger = logging.getLogger("dfn_cave_studio")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = Path(config.logging.file_path)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (e.g. 'dfn', 'voxel.connectivity').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"dfn_cave_studio.{name}")
