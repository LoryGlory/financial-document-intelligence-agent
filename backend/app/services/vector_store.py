"""VectorStore protocol / stub — full implementation in a future task."""

from typing import Any, Protocol


class VectorStore(Protocol):
    """Minimal interface required by the Extractor."""

    def get_all_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Return all chunks stored for *document_id*."""
        ...
