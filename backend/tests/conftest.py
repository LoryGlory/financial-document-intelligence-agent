# conftest.py — shared pytest fixtures live here
# Each fixture is a function decorated with @pytest.fixture.
# Fixtures declared here are automatically available to ALL test files
# in the tests/ directory without needing to import them.


import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure required environment variables are set for all tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-unit-tests")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
