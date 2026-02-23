from dataclasses import dataclass
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Embeddings, Where

from app.models.document import Chunk

COLLECTION_NAME = "documents"


@dataclass
class QueryResult:
    section_title: str
    excerpt: str
    score: float


class VectorStore:
    def __init__(
        self,
        client: ClientAPI | None = None,
        persist_path: str = "./chroma_db",
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = chromadb.PersistentClient(path=persist_path)

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=cast(Embeddings, embeddings),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "section_title": c.section_title,
                    "section_type": c.section_type,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        document_id: str,
        top_k: int = 5,
    ) -> list[QueryResult]:
        where: Where = cast(Where, {"document_id": {"$eq": document_id}})
        results = self._collection.query(
            query_embeddings=cast(Embeddings, [query_embedding]),
            n_results=top_k,
            where=where,
        )
        query_results: list[QueryResult] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        for doc, meta, dist in zip(docs[0], metas[0], dists[0], strict=True):
            score = 1.0 / (1.0 + dist)
            query_results.append(
                QueryResult(
                    section_title=str(meta.get("section_title", "")),
                    excerpt=doc,
                    score=round(score, 4),
                )
            )
        return query_results

    def get_all_chunks(self, document_id: str) -> list[dict[str, Any]]:
        where: Where = cast(Where, {"document_id": {"$eq": document_id}})
        results = self._collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        chunks = []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        for doc, meta in zip(docs, metas, strict=True):
            chunks.append({"text": doc, "metadata": meta})
        return chunks

    def delete_document(self, document_id: str) -> None:
        where: Where = cast(Where, {"document_id": {"$eq": document_id}})
        self._collection.delete(where=where)
