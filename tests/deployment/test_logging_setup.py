import logging

import config
from deployment.logging_setup import LOGGER_NAME, configure_logging


def test_configure_logging_creates_log_dir_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()

    configure_logging(log_filename="test.log")
    logging.getLogger(LOGGER_NAME).info("hello")
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()

    assert (tmp_path / "logs" / "test.log").exists()
    assert "hello" in (tmp_path / "logs" / "test.log").read_text()


def test_configure_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    logging.getLogger(LOGGER_NAME).handlers.clear()

    logger1 = configure_logging(log_filename="test.log")
    n_handlers_after_first = len(logger1.handlers)
    logger2 = configure_logging(log_filename="test.log")

    assert logger1 is logger2
    assert len(logger2.handlers) == n_handlers_after_first
