import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.config import get_settings
from app.dependencies import get_embedder, get_vector_store
from app.models.document import DocumentStatus, IngestResponse
from app.services.embedder import Embedder
from app.services.pdf_processor import PdfProcessor
from app.services.vector_store import VectorStore

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    settings = get_settings()
    document_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        contents = file.file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        processor = PdfProcessor(
            max_chunk_tokens=settings.max_chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        chunks = processor.process(tmp_path, document_id)

        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)
        vector_store.add_chunks(chunks, embeddings)
        sections_found = list(dict.fromkeys(c.section_title for c in chunks))

        return IngestResponse(
            document_id=document_id,
            filename=file.filename or "unknown.pdf",
            status=DocumentStatus.READY,
            chunks_count=len(chunks),
            sections_found=sections_found,
        )
    finally:
        os.unlink(tmp_path)
