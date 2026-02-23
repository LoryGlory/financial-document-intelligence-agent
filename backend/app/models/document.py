from enum import StrEnum

from pydantic import BaseModel


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Chunk(BaseModel):
    """One section-chunk stored in ChromaDB."""

    id: str
    document_id: str
    section_title: str
    section_type: str
    text: str
    page_number: int
    chunk_index: int


class IngestResponse(BaseModel):
    """Returned after a successful PDF ingest."""

    document_id: str
    filename: str
    status: DocumentStatus
    chunks_count: int
    sections_found: list[str]
