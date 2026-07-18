"""
Centralized logging with rotation. Every module in the pipeline should log
through logging.getLogger("options_pricing_engine") (or a child logger of
it, e.g. getLogger("options_pricing_engine.factors.vol_richness")) so
configure_logging() applied once at process start controls output for the
whole pipeline.

Rotation: RotatingFileHandler caps each log file at max_bytes and keeps
backup_count rotated copies (pipeline.log, pipeline.log.1, ...) -- without
this, a pipeline running daily via cron/systemd indefinitely would grow one
unbounded log file.
"""
import logging
from logging.handlers import RotatingFileHandler

import config

LOGGER_NAME = "options_pricing_engine"


def configure_logging(
    log_filename: str = "pipeline.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    level: int = logging.INFO,
) -> logging.Logger:
    """Idempotent -- calling this more than once (e.g. across test runs or
    re-imports) does not add duplicate handlers to the logger."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(
        config.LOG_DIR / log_filename, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
