"""Stub for Extractor – full implementation lives in a separate PR."""

import anthropic

from app.models.financial import ExtractionResponse
from app.services.vector_store import VectorStore


class Extractor:
    def __init__(
        self,
        vector_store: VectorStore,
        anthropic_client: anthropic.Anthropic,
    ) -> None:
        self.vector_store = vector_store
        self.anthropic_client = anthropic_client

    def extract(self, document_id: str) -> ExtractionResponse:
        raise NotImplementedError
