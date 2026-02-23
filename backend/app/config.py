from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required — no default, app fails loudly at startup if missing
    anthropic_api_key: str

    # Optional with sensible defaults
    chroma_persist_path: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-sonnet-4-6"
    top_k_chunks: int = 5
    max_chunk_tokens: int = 800
    chunk_overlap_tokens: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
