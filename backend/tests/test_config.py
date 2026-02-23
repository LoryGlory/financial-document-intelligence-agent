from app.config import Settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/tmp/chroma")
    settings = Settings()
    assert settings.anthropic_api_key == "test-key-123"
    assert settings.chroma_persist_path == "/tmp/chroma"


def test_settings_has_sensible_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.top_k_chunks == 5
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.claude_model == "claude-sonnet-4-6"
