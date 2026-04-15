# ============================================================
# utils/logger.py
# Centralised logging with loguru.
# All modules import `logger` from here — one config, everywhere.
# ============================================================

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "logs", level: str = "DEBUG") -> None:
    """
    Configure loguru for both console and rotating file output.

    Args:
        log_dir: Directory where log files are stored.
        level:   Minimum log level to capture.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove the default handler so we control all formatting
    logger.remove()

    # ── Console handler — coloured, human-readable ────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ── File handler — JSON-structured for log aggregators ───────────────
    logger.add(
        Path(log_dir) / "app_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="10 MB",      # rotate when file exceeds 10 MB
        retention="14 days",   # keep last two weeks of logs
        compression="zip",
        serialize=True,        # write as JSON lines
    )


# Initialise on first import so every module that does
#   `from utils.logger import logger`
# gets a fully configured logger instance.
setup_logger()

__all__ = ["logger", "setup_logger"]
