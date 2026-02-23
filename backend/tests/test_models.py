from app.models.document import Chunk, DocumentStatus, IngestResponse
from app.models.financial import (
    Citation,
    ExtractionResponse,
    FinancialMetrics,
    QueryResponse,
)


def test_ingest_response_requires_document_id():
    response = IngestResponse(
        document_id="abc-123",
        filename="test.pdf",
        status=DocumentStatus.READY,
        chunks_count=5,
        sections_found=["MD&A", "Risk Factors"],
    )
    assert response.document_id == "abc-123"
    assert response.status == DocumentStatus.READY


def test_chunk_has_required_metadata():
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        section_title="Risk Factors",
        section_type="risk_factors",
        text="The company faces various risks...",
        page_number=12,
        chunk_index=0,
    )
    assert chunk.section_title == "Risk Factors"
    assert chunk.page_number == 12


def test_query_response_has_citations():
    response = QueryResponse(
        answer="Revenue grew 8% driven by iPhone sales.",
        citations=[
            Citation(
                section_title="MD&A",
                excerpt="iPhone revenue increased 6%...",
                score=0.91,
            )
        ],
    )
    assert len(response.citations) == 1
    assert response.citations[0].score == 0.91


def test_financial_metrics_allows_null_fields():
    metrics = FinancialMetrics(
        revenue=None,
        eps=None,
        net_income=None,
        gross_margin=None,
        guidance=None,
        yoy_deltas=None,
    )
    assert metrics.revenue is None


def test_extraction_response_structure():
    response = ExtractionResponse(
        document_id="doc-1",
        company_name="Apple Inc.",
        fiscal_year="2024",
        filing_type="10-K",
        metrics=FinancialMetrics(),
    )
    assert response.company_name == "Apple Inc."
