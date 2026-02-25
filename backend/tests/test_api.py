import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_extractor, get_rag_pipeline, get_vector_store
from app.main import app
from app.models.document import Chunk
from app.models.financial import (
    Citation,
    ExtractionResponse,
    FinancialMetrics,
    QueryResponse,
    RevenueMetric,
)


@pytest.fixture
def mock_embedder():
    m = MagicMock()
    m.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    return m


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def mock_rag_pipeline():
    m = MagicMock()
    m.query.return_value = QueryResponse(
        answer="Revenue grew 8% [1].",
        citations=[Citation(section_title="MD&A", excerpt="Revenue grew 8%", score=0.9)],
    )
    return m


@pytest.fixture
def mock_extractor():
    m = MagicMock()
    m.extract.return_value = ExtractionResponse(
        document_id="doc-1",
        company_name="Apple Inc.",
        fiscal_year="2024",
        filing_type="10-K",
        metrics=FinancialMetrics(revenue=RevenueMetric(value=391.0)),
    )
    return m


@pytest.fixture
def client(mock_embedder, mock_vector_store, mock_rag_pipeline, mock_extractor):
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag_pipeline
    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_returns_document_id(client):
    with patch("app.api.ingest.PdfProcessor") as MockProcessor:
        mock_proc = MagicMock()
        mock_proc.process.return_value = [
            Chunk(
                id="c1",
                document_id="doc-1",
                section_title="MD&A",
                section_type="mda",
                text="Revenue...",
                page_number=1,
                chunk_index=0,
            )
        ]
        MockProcessor.return_value = mock_proc

        pdf_bytes = b"%PDF-1.4 fake pdf content"
        response = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "ready"
    assert data["chunks_count"] == 1


def test_ingest_rejects_non_pdf(client):
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_query_returns_answer_and_citations(client):
    response = client.post(
        "/query",
        json={"document_id": "doc-1", "question": "What drove revenue?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["citations"]) == 1
    assert data["citations"][0]["section_title"] == "MD&A"


def test_extract_returns_metrics(client):
    response = client.post("/extract", json={"document_id": "doc-1"})
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Apple Inc."
    assert data["metrics"]["revenue"]["value"] == 391.0
