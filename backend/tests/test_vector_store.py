import chromadb
import pytest

from app.models.document import Chunk
from app.services.vector_store import VectorStore


@pytest.fixture
def store():
    client = chromadb.EphemeralClient()
    return VectorStore(client=client)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="chunk-1",
            document_id="doc-abc",
            section_title="Risk Factors",
            section_type="risk_factors",
            text="The company faces intense competition.",
            page_number=5,
            chunk_index=0,
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-abc",
            section_title="MD&A",
            section_type="mda",
            text="Revenue increased 8% year over year.",
            page_number=20,
            chunk_index=1,
        ),
    ]


def test_add_and_query_chunks(store, sample_chunks):
    embeddings = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]
    store.add_chunks(sample_chunks, embeddings)
    results = store.query(
        query_embedding=[0.1, 0.2, 0.3],
        document_id="doc-abc",
        top_k=2,
    )
    assert len(results) >= 1
    assert results[0].section_title in ("Risk Factors", "MD&A")


def test_query_filters_by_document_id(store, sample_chunks):
    embeddings = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]
    store.add_chunks(sample_chunks, embeddings)
    results = store.query(
        query_embedding=[0.1, 0.2, 0.3],
        document_id="doc-OTHER",
        top_k=2,
    )
    assert len(results) == 0


def test_get_all_chunks_for_document(store, sample_chunks):
    embeddings = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]
    store.add_chunks(sample_chunks, embeddings)
    chunks = store.get_all_chunks(document_id="doc-abc")
    assert len(chunks) == 2


def test_delete_document(store, sample_chunks):
    embeddings = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]
    store.add_chunks(sample_chunks, embeddings)
    store.delete_document("doc-abc")
    chunks = store.get_all_chunks(document_id="doc-abc")
    assert len(chunks) == 0
