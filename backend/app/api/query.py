from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_rag_pipeline
from app.models.financial import QueryResponse
from app.services.rag_pipeline import RagPipeline

router = APIRouter()


class QueryRequest(BaseModel):
    document_id: str
    question: str


@router.post("/query", response_model=QueryResponse)
def query_document(
    request: QueryRequest,
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    return pipeline.query(
        document_id=request.document_id,
        question=request.question,
    )
