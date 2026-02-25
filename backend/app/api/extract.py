from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_extractor
from app.models.financial import ExtractionResponse
from app.services.extractor import Extractor

router = APIRouter()


class ExtractRequest(BaseModel):
    document_id: str


@router.post("/extract", response_model=ExtractionResponse)
def extract_metrics(
    request: ExtractRequest,
    extractor: Extractor = Depends(get_extractor),
) -> ExtractionResponse:
    return extractor.extract(document_id=request.document_id)
