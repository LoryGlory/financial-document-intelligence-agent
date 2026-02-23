"""Stub for VectorStore – full implementation lives in a separate PR."""

from app.models.document import Chunk


class QueryResult:
    """A single retrieved chunk from the vector store."""

    def __init__(self, section_title: str, excerpt: str, score: float) -> None:
        self.section_title = section_title
        self.excerpt = excerpt
        self.score = score


class VectorStore:
    def __init__(self, persist_path: str = "./chroma_db") -> None:
        self.persist_path = persist_path

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    def query(
        self,
        query_embedding: list[float],
        document_id: str,
        top_k: int = 5,
    ) -> list[QueryResult]:
        raise NotImplementedError
