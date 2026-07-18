"""Shared test fixtures: every test gets an isolated, empty cache DB so tests
never share state or touch a real developer's accumulated cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DB_PATH", tmp_path / "test_cache.sqlite")
    yield
