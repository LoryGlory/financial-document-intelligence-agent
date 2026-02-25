import json
from unittest.mock import MagicMock

import pytest

from app.models.financial import ExtractionResponse
from app.services.extractor import Extractor


@pytest.fixture
def mock_anthropic():
    client = MagicMock()
    message = MagicMock()
    message.content = [
        MagicMock(
            text=json.dumps(
                {
                    "company_name": "Apple Inc.",
                    "fiscal_year": "2024",
                    "filing_type": "10-K",
                    "metrics": {
                        "revenue": {"value": 391.0, "unit": "USD_billions", "period": "FY2024"},
                        "eps": {"value": 6.11, "diluted": True},
                        "net_income": {"value": 93.7, "unit": "USD_billions"},
                        "gross_margin": {"value": 46.2},
                        "guidance": None,
                        "yoy_deltas": {"revenue": 2.0, "eps": 10.9, "net_income": 3.4},
                    },
                }
            )
        )
    ]
    client.messages.create.return_value = message
    return client


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.get_all_chunks.return_value = [
        {"text": "Revenue was $391 billion in FY2024.", "metadata": {"chunk_index": 0}},
        {"text": "EPS was $6.11 diluted.", "metadata": {"chunk_index": 1}},
    ]
    return store


def test_extractor_returns_extraction_response(mock_anthropic, mock_vector_store):
    extractor = Extractor(
        vector_store=mock_vector_store,
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
    )
    result = extractor.extract(document_id="doc-1")
    assert isinstance(result, ExtractionResponse)
    assert result.company_name == "Apple Inc."


def test_extractor_parses_metrics(mock_anthropic, mock_vector_store):
    extractor = Extractor(
        vector_store=mock_vector_store,
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
    )
    result = extractor.extract(document_id="doc-1")
    assert result.metrics.revenue is not None
    assert result.metrics.revenue.value == 391.0
    assert result.metrics.guidance is None


def test_extractor_handles_invalid_json(mock_vector_store):
    bad_client = MagicMock()
    bad_client.messages.create.return_value.content = [
        MagicMock(text="Sorry, I cannot extract that information.")
    ]
    extractor = Extractor(
        vector_store=mock_vector_store,
        anthropic_client=bad_client,
        model="claude-sonnet-4-6",
    )
    result = extractor.extract(document_id="doc-1")
    assert isinstance(result, ExtractionResponse)
    assert result.company_name is None
