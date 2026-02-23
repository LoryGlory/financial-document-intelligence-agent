from unittest.mock import MagicMock

import pytest

from app.services.rag_pipeline import RagPipeline
from app.services.vector_store import QueryResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.query.return_value = [
        QueryResult(
            section_title="MD&A",
            excerpt="Revenue increased 8% to $391 billion.",
            score=0.92,
        ),
        QueryResult(
            section_title="Risk Factors",
            excerpt="Competition from Android manufacturers remains intense.",
            score=0.78,
        ),
    ]
    return store


@pytest.fixture
def mock_anthropic():
    client = MagicMock()
    message = MagicMock()
    message.content = [
        MagicMock(text="Revenue grew 8% driven by iPhone [1]. Competition is a risk [2].")
    ]
    client.messages.create.return_value = message
    return client


def test_rag_pipeline_returns_answer(mock_embedder, mock_vector_store, mock_anthropic):
    pipeline = RagPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
        top_k=5,
    )
    response = pipeline.query(document_id="doc-1", question="What drove revenue growth?")
    assert "Revenue" in response.answer or "grew" in response.answer


def test_rag_pipeline_returns_citations(mock_embedder, mock_vector_store, mock_anthropic):
    pipeline = RagPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
        top_k=5,
    )
    response = pipeline.query(document_id="doc-1", question="What drove revenue?")
    assert len(response.citations) == 2
    assert response.citations[0].section_title == "MD&A"
    assert response.citations[0].score == 0.92


def test_prompt_contains_numbered_context(mock_embedder, mock_vector_store, mock_anthropic):
    pipeline = RagPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
        top_k=5,
    )
    pipeline.query(document_id="doc-1", question="Revenue?")
    call_kwargs = mock_anthropic.messages.create.call_args
    prompt = call_kwargs[1]["messages"][0]["content"]
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_rag_returns_empty_when_no_chunks(mock_embedder, mock_anthropic):
    empty_store = MagicMock()
    empty_store.query.return_value = []
    pipeline = RagPipeline(
        embedder=mock_embedder,
        vector_store=empty_store,
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
        top_k=5,
    )
    response = pipeline.query(document_id="doc-1", question="Revenue?")
    assert response.citations == []
    mock_anthropic.messages.create.assert_not_called()
