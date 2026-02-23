"""Stub for RagPipeline – full implementation lives in a separate PR."""

import anthropic

from app.models.financial import QueryResponse
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        anthropic_client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.anthropic_client = anthropic_client
        self.model = model
        self.top_k = top_k

    def query(self, document_id: str, question: str) -> QueryResponse:
        raise NotImplementedError
