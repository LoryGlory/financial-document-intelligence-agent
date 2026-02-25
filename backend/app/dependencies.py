from functools import lru_cache

import anthropic

from app.config import get_settings
from app.services.embedder import Embedder
from app.services.extractor import Extractor
from app.services.rag_pipeline import RagPipeline
from app.services.vector_store import VectorStore


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(model_name=settings.embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(persist_path=settings.chroma_persist_path)


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    settings = get_settings()
    return RagPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        anthropic_client=get_anthropic_client(),
        model=settings.claude_model,
        top_k=settings.top_k_chunks,
    )


@lru_cache
def get_extractor() -> Extractor:
    return Extractor(
        vector_store=get_vector_store(),
        anthropic_client=get_anthropic_client(),
    )
