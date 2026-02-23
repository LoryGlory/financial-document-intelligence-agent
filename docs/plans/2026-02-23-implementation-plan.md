# Financial Document Intelligence Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a portfolio-grade RAG-powered Q&A system for financial documents with a polished Next.js UI, FastAPI backend, ChromaDB vector store, and Anthropic claude-sonnet-4-6.

**Architecture:** Section-aware PDF chunking → MiniLM-L6 embeddings → ChromaDB retrieval → grounded Claude answers with inline citations. Separate `/extract` endpoint returns structured JSON metrics via Claude + Pydantic.

**Tech Stack:** Python 3.12, FastAPI, pdfplumber, sentence-transformers, ChromaDB, Anthropic SDK, Next.js 14 App Router, shadcn/ui, Tailwind CSS, pytest, ruff, mypy, Docker, GitHub Actions, Railway.

---

## Branch convention
Every task runs on its own branch. Branch off `main`. Name: `feat/<task>`, `chore/<task>`, `docs/<task>`.
Never commit directly to `main`.

---

## Phase 1 — Backend Foundation

---

### Task 1: Project scaffold + pyproject.toml

**Branch:** `chore/backend-scaffold`

**Why this matters:** `pyproject.toml` is the modern Python project file. It replaces `setup.py`, `setup.cfg`, and `requirements.txt`. We configure `ruff` (linter/formatter) and `mypy` (type checker) here so they apply project-wide.

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p backend/app/api backend/app/services backend/app/models backend/tests
touch backend/app/__init__.py backend/app/api/__init__.py
touch backend/app/services/__init__.py backend/app/models/__init__.py
touch backend/tests/__init__.py
```

**Step 2: Write `backend/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "financial-document-intelligence"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "anthropic>=0.40.0",
    "pdfplumber>=0.11.0",
    "sentence-transformers>=3.3.0",
    "chromadb>=0.5.23",
    "python-multipart>=0.0.12",
    "httpx>=0.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=app --cov-report=term-missing --cov-fail-under=80"

[tool.coverage.run]
omit = ["tests/*", "*/migrations/*"]
```

**Step 3: Write `backend/tests/conftest.py`** (empty for now, we'll add fixtures per task)

```python
# conftest.py — shared pytest fixtures live here
# Each fixture is a function decorated with @pytest.fixture.
# Fixtures declared here are automatically available to ALL test files
# in the tests/ directory without needing to import them.
```

**Step 4: Write `.gitignore`**

```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
dist/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage

# ChromaDB
chroma_db/

# Environment
.env
.env.local
.env.*.local

# Next.js
frontend/node_modules/
frontend/.next/
frontend/out/

# OS
.DS_Store
```

**Step 5: Install deps and verify tooling works**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check app/     # Expected: no output (nothing to lint yet)
mypy app/           # Expected: Success: no issues found
```

**Step 6: Commit**

```bash
git add backend/pyproject.toml backend/app/ backend/tests/ .gitignore
git commit -m "chore: scaffold backend project structure and pyproject.toml"
```

---

### Task 2: Pydantic models

**Branch:** `feat/pydantic-models`

**Why this matters:** Pydantic models are the contract between layers. They validate data coming in and out of the API. Because we define them first (TDD-style), every downstream service knows exactly what shape of data to produce. `Optional[X]` fields default to `None` — critical for extraction where not all metrics appear in every filing.

**Files:**
- Create: `backend/app/models/document.py`
- Create: `backend/app/models/financial.py`
- Create: `backend/tests/test_models.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_models.py
import pytest
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
    # Not every filing has guidance — null fields must be allowed
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
```

**Step 2: Run tests — verify they FAIL**

```bash
cd backend && pytest tests/test_models.py -v
# Expected: ImportError — modules don't exist yet
```

**Step 3: Write `backend/app/models/document.py`**

```python
# document.py — data shapes for the ingestion pipeline
#
# Pydantic v2 uses `model_config` instead of inner `class Config`.
# `str` enum inherits from both str and Enum so FastAPI can serialize it
# as a plain string in JSON responses (not {"value": "ready"}).

from enum import Enum
from pydantic import BaseModel


class DocumentStatus(str, Enum):
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
```

**Step 4: Write `backend/app/models/financial.py`**

```python
# financial.py — data shapes for query and extraction responses
#
# Optional[X] with default None means the field is not required.
# This is important for extraction: if a filing doesn't mention
# guidance, we return null rather than hallucinating a value.

from typing import Optional
from pydantic import BaseModel


class Citation(BaseModel):
    """One retrieved chunk used to ground an answer."""
    section_title: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    """Answer + supporting citations from the RAG pipeline."""
    answer: str
    citations: list[Citation]


class RevenueMetric(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None


class EpsMetric(BaseModel):
    value: Optional[float] = None
    diluted: Optional[bool] = None


class NetIncomeMetric(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None


class GrossMarginMetric(BaseModel):
    value: Optional[float] = None


class GuidanceMetric(BaseModel):
    revenue_low: Optional[float] = None
    revenue_high: Optional[float] = None
    period: Optional[str] = None


class YoyDeltas(BaseModel):
    revenue: Optional[float] = None
    eps: Optional[float] = None
    net_income: Optional[float] = None


class FinancialMetrics(BaseModel):
    """All extractable metrics. Every field is Optional — not all filings
    contain all metrics. Claude returns null for unavailable fields."""
    revenue: Optional[RevenueMetric] = None
    eps: Optional[EpsMetric] = None
    net_income: Optional[NetIncomeMetric] = None
    gross_margin: Optional[GrossMarginMetric] = None
    guidance: Optional[GuidanceMetric] = None
    yoy_deltas: Optional[YoyDeltas] = None


class ExtractionResponse(BaseModel):
    """Structured metrics extracted from a financial filing."""
    document_id: str
    company_name: Optional[str] = None
    fiscal_year: Optional[str] = None
    filing_type: Optional[str] = None
    metrics: FinancialMetrics = FinancialMetrics()
```

**Step 5: Run tests — verify they PASS**

```bash
pytest tests/test_models.py -v
# Expected: 5 passed
```

**Step 6: Lint + type-check**

```bash
ruff check app/models/
mypy app/models/
# Expected: no errors
```

**Step 7: Commit**

```bash
git add backend/app/models/ backend/tests/test_models.py
git commit -m "feat: add Pydantic models for document and financial data shapes"
```

---

### Task 3: Config with pydantic-settings

**Branch:** `feat/config`

**Why this matters:** `pydantic-settings` reads values from environment variables (and `.env` files) and validates them as typed Python objects. This means if `ANTHROPIC_API_KEY` is missing, the app fails at startup with a clear error — not silently at request time.

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/.env.example`
- Create: `backend/tests/test_config.py`

**Step 1: Write failing test**

```python
# backend/tests/test_config.py
import os
import pytest
from app.config import Settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/tmp/chroma")
    settings = Settings()
    assert settings.anthropic_api_key == "test-key-123"
    assert settings.chroma_persist_path == "/tmp/chroma"


def test_settings_has_sensible_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.top_k_chunks == 5
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.claude_model == "claude-sonnet-4-6"
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_config.py -v
# Expected: ImportError
```

**Step 3: Write `backend/app/config.py`**

```python
# config.py — central configuration via environment variables
#
# pydantic-settings automatically reads from:
#   1. Environment variables (highest priority)
#   2. .env file (if present)
#   3. Default values defined here
#
# The `@lru_cache` on `get_settings()` means the Settings object is
# created once and reused — important because sentence-transformers
# loads a model on first use, which is slow.

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required — no default, app fails loudly if missing
    anthropic_api_key: str

    # Optional with sensible defaults
    chroma_persist_path: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-sonnet-4-6"
    top_k_chunks: int = 5
    max_chunk_tokens: int = 800
    chunk_overlap_tokens: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Step 4: Write `backend/.env.example`**

```bash
# Copy to .env and fill in real values
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_PATH=./chroma_db
```

**Step 5: Run — verify PASS**

```bash
pytest tests/test_config.py -v
# Expected: 2 passed
```

**Step 6: Commit**

```bash
git add backend/app/config.py backend/.env.example backend/tests/test_config.py
git commit -m "feat: add pydantic-settings config with env var validation"
```

---

### Task 4: PDF Processor service

**Branch:** `feat/pdf-processor`

**Why this matters:** This is the most domain-specific code in the project. 10-K filings have a standard structure mandated by the SEC: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements), etc. We detect these with regex and use them as natural chunk boundaries. This dramatically improves retrieval quality compared to arbitrary token windows.

**Files:**
- Create: `backend/app/services/pdf_processor.py`
- Create: `backend/tests/test_pdf_processor.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_pdf_processor.py
#
# We don't need real PDFs to test the processor — we test the
# text processing logic directly. Real PDF parsing is tested
# via integration tests with a tiny sample PDF.

import pytest
from app.services.pdf_processor import PdfProcessor, Section


def test_detect_10k_sections():
    processor = PdfProcessor(max_chunk_tokens=800, overlap_tokens=100)
    text = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION

Item 1. Business
Apple Inc. designs, manufactures and markets smartphones.

Item 1A. Risk Factors
The company operates in a highly competitive market.

Item 7. Management's Discussion and Analysis
Revenue for fiscal 2024 was $391 billion.
"""
    sections = processor.detect_sections(text)
    titles = [s.title for s in sections]
    assert "Item 1. Business" in titles
    assert "Item 1A. Risk Factors" in titles
    assert "Item 7. Management's Discussion and Analysis" in titles


def test_detect_sections_returns_content():
    processor = PdfProcessor(max_chunk_tokens=800, overlap_tokens=100)
    text = """
Item 1. Business
Apple Inc. designs smartphones.

Item 1A. Risk Factors
Competition is intense.
"""
    sections = processor.detect_sections(text)
    business = next(s for s in sections if "Business" in s.title)
    assert "Apple Inc." in business.content


def test_chunk_short_section_returns_one_chunk():
    processor = PdfProcessor(max_chunk_tokens=800, overlap_tokens=100)
    section = Section(title="Risk Factors", content="Short content.", page_number=5)
    chunks = processor.chunk_section(section, document_id="doc-1", chunk_offset=0)
    assert len(chunks) == 1
    assert chunks[0].section_title == "Risk Factors"
    assert chunks[0].document_id == "doc-1"


def test_chunk_long_section_produces_multiple_chunks():
    processor = PdfProcessor(max_chunk_tokens=50, overlap_tokens=10)
    # 200 words ~ 267 tokens, should produce multiple chunks at 50-token limit
    long_content = " ".join(["word"] * 200)
    section = Section(title="MD&A", content=long_content, page_number=10)
    chunks = processor.chunk_section(section, document_id="doc-1", chunk_offset=0)
    assert len(chunks) > 1


def test_chunks_have_sequential_indices():
    processor = PdfProcessor(max_chunk_tokens=50, overlap_tokens=10)
    long_content = " ".join(["word"] * 200)
    section = Section(title="MD&A", content=long_content, page_number=10)
    chunks = processor.chunk_section(section, document_id="doc-1", chunk_offset=0)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_pdf_processor.py -v
# Expected: ImportError
```

**Step 3: Write `backend/app/services/pdf_processor.py`**

```python
# pdf_processor.py — PDF text extraction and section-aware chunking
#
# WHY SECTION-AWARE CHUNKING:
# Financial filings have a known structure (SEC mandates Item numbers).
# By detecting section headers, we keep semantically related content
# together. A question about "risk factors" retrieves the Risk Factors
# section, not an arbitrary 512-token window that might straddle two topics.
#
# SUB-CHUNKING:
# Some sections (MD&A) can be 50+ pages. We split those at ~800 tokens
# with 100-token overlap. The overlap prevents a sentence at a chunk
# boundary from losing context.

import re
import uuid
from dataclasses import dataclass, field

import pdfplumber

from app.models.document import Chunk


# Regex patterns for SEC 10-K section headers.
# Covers: "Item 1.", "Item 1A.", "ITEM 7.", etc.
# The (?i) flag makes it case-insensitive.
SECTION_PATTERNS = [
    r"(?i)^item\s+\d+[a-z]?\.\s+.+",          # Standard: "Item 1. Business"
    r"(?i)^management.s discussion",             # Alternate MD&A header
    r"(?i)^notes to (consolidated )?financial",  # Notes section
    r"(?i)^selected financial data",
    r"(?i)^quantitative and qualitative",
    r"(?i)^controls and procedures",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE)


@dataclass
class Section:
    title: str
    content: str
    page_number: int


@dataclass
class PdfProcessor:
    max_chunk_tokens: int = 800
    overlap_tokens: int = 100

    def extract_text(self, pdf_path: str) -> tuple[str, dict[int, int]]:
        """Extract full text from a PDF. Returns (text, page_char_map).

        page_char_map maps character offset → page number, so we can
        recover page numbers when we later split into sections.
        """
        full_text = ""
        page_char_map: dict[int, int] = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                page_char_map[len(full_text)] = page_num
                full_text += page_text + "\n"

        return full_text, page_char_map

    def detect_sections(self, text: str) -> list[Section]:
        """Split text into named sections using SEC header patterns.

        Works by finding all header matches, then treating the text
        between consecutive headers as that section's content.
        """
        matches = list(SECTION_REGEX.finditer(text))

        if not matches:
            # No headers found — treat entire text as one section
            return [Section(title="Document", content=text.strip(), page_number=1)]

        sections: list[Section] = []
        for i, match in enumerate(matches):
            title = match.group(0).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if content:  # Skip empty sections
                sections.append(Section(title=title, content=content, page_number=1))

        return sections

    def chunk_section(
        self, section: Section, document_id: str, chunk_offset: int
    ) -> list[Chunk]:
        """Split a section into chunks of at most max_chunk_tokens tokens.

        We approximate tokens as words / 0.75 (1 token ≈ 0.75 words on average).
        A proper implementation would use a tokenizer, but word-based
        approximation is fast and accurate enough for retrieval quality.
        """
        words = section.content.split()
        approx_max_words = int(self.max_chunk_tokens * 0.75)
        approx_overlap_words = int(self.overlap_tokens * 0.75)

        if len(words) <= approx_max_words:
            return [
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    section_title=section.title,
                    section_type=self._classify_section(section.title),
                    text=section.content,
                    page_number=section.page_number,
                    chunk_index=chunk_offset,
                )
            ]

        chunks: list[Chunk] = []
        step = max(1, approx_max_words - approx_overlap_words)
        chunk_idx = 0

        for start in range(0, len(words), step):
            chunk_words = words[start : start + approx_max_words]
            if not chunk_words:
                break
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    section_title=section.title,
                    section_type=self._classify_section(section.title),
                    text=" ".join(chunk_words),
                    page_number=section.page_number,
                    chunk_index=chunk_offset + chunk_idx,
                )
            )
            chunk_idx += 1

        return chunks

    def process(self, pdf_path: str, document_id: str) -> list[Chunk]:
        """Full pipeline: PDF → text → sections → chunks."""
        text, _ = self.extract_text(pdf_path)
        sections = self.detect_sections(text)

        all_chunks: list[Chunk] = []
        chunk_offset = 0
        for section in sections:
            chunks = self.chunk_section(section, document_id, chunk_offset)
            all_chunks.extend(chunks)
            chunk_offset += len(chunks)

        return all_chunks

    @staticmethod
    def _classify_section(title: str) -> str:
        """Map a section title to a canonical type for metadata filtering."""
        title_lower = title.lower()
        if "risk factor" in title_lower:
            return "risk_factors"
        if "management" in title_lower or "md&a" in title_lower:
            return "mda"
        if "financial statement" in title_lower:
            return "financial_statements"
        if "business" in title_lower:
            return "business"
        if "note" in title_lower:
            return "notes"
        return "other"
```

**Step 4: Run — verify PASS**

```bash
pytest tests/test_pdf_processor.py -v
# Expected: 5 passed
```

**Step 5: Commit**

```bash
git add backend/app/services/pdf_processor.py backend/tests/test_pdf_processor.py
git commit -m "feat: add section-aware PDF processor with sub-chunking"
```

---

### Task 5: Embedder service

**Branch:** `feat/embedder`

**Why this matters:** The embedder converts text → dense vector (a list of 384 floats for MiniLM-L6). Semantically similar text produces similar vectors, which is how ChromaDB finds relevant chunks for a question. Loading the model is slow (~1–2s), so we load it once at startup and reuse it via `@lru_cache`.

**Files:**
- Create: `backend/app/services/embedder.py`
- Create: `backend/tests/test_embedder.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_embedder.py
#
# We don't call the real model in tests — that would:
# 1. Make tests slow (model download + inference)
# 2. Require a GPU or specific hardware in CI
# Instead, we mock the SentenceTransformer class.

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from app.services.embedder import Embedder


@pytest.fixture
def mock_model():
    """Patch SentenceTransformer so no real model is loaded."""
    with patch("app.services.embedder.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        # encode() returns a numpy array of floats
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_instance
        yield mock_cls


def test_embed_single_text(mock_model):
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed("What is Apple's revenue?")
    assert isinstance(result, list)
    assert len(result) == 3  # our mock returns 3-dim vector


def test_embed_batch(mock_model):
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    mock_model.return_value.encode.return_value = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])
    results = embedder.embed_batch(["text one", "text two"])
    assert len(results) == 2
    assert len(results[0]) == 3
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_embedder.py -v
```

**Step 3: Write `backend/app/services/embedder.py`**

```python
# embedder.py — sentence-transformers wrapper
#
# SentenceTransformer.encode() returns a numpy array.
# We convert to a plain Python list[float] because:
# - ChromaDB accepts list[float]
# - JSON serialisation works automatically
# - No numpy dependency leaks into callers

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # This loads the model weights from disk (or downloads them).
        # In production, the model is cached at container startup.
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single string → 384-dimensional vector."""
        vector = self._model.encode([text])
        return vector[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one forward pass (faster than looping)."""
        vectors = self._model.encode(texts)
        return [v.tolist() for v in vectors]
```

**Step 4: Run — verify PASS**

```bash
pytest tests/test_embedder.py -v
# Expected: 2 passed
```

**Step 5: Commit**

```bash
git add backend/app/services/embedder.py backend/tests/test_embedder.py
git commit -m "feat: add sentence-transformers embedder service"
```

---

### Task 6: Vector store service (ChromaDB)

**Branch:** `feat/vector-store`

**Why this matters:** ChromaDB is our search engine. It stores each chunk as a (vector, text, metadata) triple. The `where` filter on `document_id` ensures queries only search within the user's selected document. We wrap ChromaDB behind our own `VectorStore` class so the rest of the app never imports `chromadb` directly — easy to swap later.

**Files:**
- Create: `backend/app/services/vector_store.py`
- Create: `backend/tests/test_vector_store.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_vector_store.py
#
# ChromaDB supports an in-memory client (EphemeralClient) for testing.
# No disk writes, no cleanup needed. Perfect for CI.

import pytest
import chromadb
from app.services.vector_store import VectorStore
from app.models.document import Chunk


@pytest.fixture
def store():
    """In-memory ChromaDB instance — no state persists between tests."""
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

    # Query for a different document — should return nothing
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
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_vector_store.py -v
```

**Step 3: Write `backend/app/services/vector_store.py`**

```python
# vector_store.py — ChromaDB wrapper
#
# We use a single collection ("documents") for all chunks.
# Each chunk's metadata carries document_id so we can filter.
#
# ChromaDB query results come back as dicts-of-lists:
#   { "ids": [["id1"]], "documents": [["text1"]], "metadatas": [[{...}]], "distances": [[0.1]] }
# The outer list is per query (we always send one query), hence [0].
#
# DISTANCE vs SIMILARITY: ChromaDB returns L2 distance by default.
# Lower = more similar. We convert to a 0–1 score with 1/(1+distance).

from dataclasses import dataclass
from typing import Any

import chromadb

from app.models.document import Chunk

COLLECTION_NAME = "documents"


@dataclass
class QueryResult:
    section_title: str
    excerpt: str
    score: float


class VectorStore:
    def __init__(self, client: chromadb.ClientAPI | None = None, persist_path: str = "./chroma_db") -> None:
        if client is not None:
            self._client = client
        else:
            self._client = chromadb.PersistentClient(path=persist_path)

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings and metadata."""
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "section_title": c.section_title,
                    "section_type": c.section_type,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def query(
        self, query_embedding: list[float], document_id: str, top_k: int = 5
    ) -> list[QueryResult]:
        """Find the top_k most relevant chunks for a question, within one document."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": {"$eq": document_id}},
        )

        query_results: list[QueryResult] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]

        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            score = 1.0 / (1.0 + dist)  # convert L2 distance to similarity score
            query_results.append(
                QueryResult(
                    section_title=str(meta.get("section_title", "")),
                    excerpt=doc,
                    score=round(score, 4),
                )
            )

        return query_results

    def get_all_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve all stored chunks for a document (used by /extract)."""
        results = self._collection.get(
            where={"document_id": {"$eq": document_id}},
            include=["documents", "metadatas"],
        )
        chunks = []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            chunks.append({"text": doc, "metadata": meta})
        return chunks

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks for a document."""
        self._collection.delete(where={"document_id": {"$eq": document_id}})
```

**Step 4: Run — verify PASS**

```bash
pytest tests/test_vector_store.py -v
# Expected: 4 passed
```

**Step 5: Commit**

```bash
git add backend/app/services/vector_store.py backend/tests/test_vector_store.py
git commit -m "feat: add ChromaDB vector store wrapper with document-scoped queries"
```

---

### Task 7: RAG pipeline service

**Branch:** `feat/rag-pipeline`

**Why this matters:** This is the intellectual core of the project. The prompt engineering pattern here — numbered context blocks + instruction to cite by number — is the standard way to get reliable citations from LLMs. The model doesn't hallucinate sources because it's literally copying from the numbered blocks we provided.

**Files:**
- Create: `backend/app/services/rag_pipeline.py`
- Create: `backend/tests/test_rag_pipeline.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_rag_pipeline.py
#
# We mock both the Embedder and the Anthropic client.
# This tests our prompt construction and response parsing logic
# without making any real API calls.

import pytest
from unittest.mock import MagicMock, AsyncMock
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
    message.content = [MagicMock(text="Revenue grew 8% driven by iPhone [1]. Competition is a risk [2].")]
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
    response = pipeline.query(
        document_id="doc-1",
        question="What drove revenue growth?",
    )
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
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_rag_pipeline.py -v
```

**Step 3: Write `backend/app/services/rag_pipeline.py`**

```python
# rag_pipeline.py — retrieval-augmented generation
#
# THE GROUNDING PATTERN:
# We number each retrieved chunk [1], [2], etc. and instruct Claude to
# cite by number. This works because:
# 1. Claude has been trained to follow instruction formats
# 2. The model literally has the source text in its context window
# 3. We can parse "[N]" references back to specific chunks
#
# This is NOT the same as RAG hallucination prevention — the model
# *could* still ignore the sources. The "only use provided context"
# instruction, combined with high-quality retrieval, keeps it honest.

import anthropic

from app.models.financial import Citation, QueryResponse
from app.services.embedder import Embedder
from app.services.vector_store import QueryResult, VectorStore

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions about
financial documents using ONLY the provided context passages.

Rules:
- Cite sources using [N] notation inline (e.g. "Revenue grew 8% [1].")
- If the answer isn't in the provided context, say "I don't have enough information to answer that."
- Be precise about numbers and dates.
- Keep answers concise and factual."""


def _build_context_block(results: list[QueryResult]) -> str:
    lines = []
    for i, result in enumerate(results, start=1):
        lines.append(f"[{i}] {result.section_title}:\n{result.excerpt}")
    return "\n\n".join(lines)


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        anthropic_client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._client = anthropic_client
        self._model = model
        self._top_k = top_k

    def query(self, document_id: str, question: str) -> QueryResponse:
        # Step 1: embed the question
        question_embedding = self._embedder.embed(question)

        # Step 2: retrieve top-k relevant chunks
        results = self._store.query(
            query_embedding=question_embedding,
            document_id=document_id,
            top_k=self._top_k,
        )

        if not results:
            return QueryResponse(
                answer="I couldn't find relevant information in this document.",
                citations=[],
            )

        # Step 3: build a numbered context block
        context = _build_context_block(results)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {question}"

        # Step 4: call Claude
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = message.content[0].text

        # Step 5: citations = all retrieved chunks (Claude cited [N] inline)
        citations = [
            Citation(
                section_title=r.section_title,
                excerpt=r.excerpt,
                score=r.score,
            )
            for r in results
        ]

        return QueryResponse(answer=answer, citations=citations)
```

**Step 4: Run — verify PASS**

```bash
pytest tests/test_rag_pipeline.py -v
# Expected: 3 passed
```

**Step 5: Commit**

```bash
git add backend/app/services/rag_pipeline.py backend/tests/test_rag_pipeline.py
git commit -m "feat: add RAG pipeline with grounded prompting and citation extraction"
```

---

### Task 8: Extractor service

**Branch:** `feat/extractor`

**Why this matters:** Structured extraction demonstrates a different AI pattern than RAG — instead of retrieval + generation, we give Claude a JSON schema and ask it to fill in only what it finds. The `response_format` / JSON-mode approach is more reliable than asking Claude to "return JSON" in a free-form prompt. Null fields signal honest absence, not hallucination.

**Files:**
- Create: `backend/app/services/extractor.py`
- Create: `backend/tests/test_extractor.py`

**Step 1: Write failing tests**

```python
# backend/tests/test_extractor.py

import json
import pytest
from unittest.mock import MagicMock
from app.services.extractor import Extractor
from app.models.financial import ExtractionResponse


@pytest.fixture
def mock_anthropic():
    client = MagicMock()
    message = MagicMock()
    # Simulate Claude returning valid JSON
    message.content = [MagicMock(text=json.dumps({
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
        }
    }))]
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
    """If Claude returns malformed JSON, we return an empty extraction."""
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
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_extractor.py -v
```

**Step 3: Write `backend/app/services/extractor.py`**

```python
# extractor.py — structured metric extraction via Claude
#
# APPROACH: We assemble all document chunks into ordered text, then
# ask Claude to extract specific fields as JSON matching our schema.
#
# We include the full JSON schema in the prompt so Claude knows exactly
# what to return. The `or null` instruction is important — without it,
# Claude tends to fabricate plausible-looking numbers for missing fields.

import json
import anthropic

from app.models.financial import ExtractionResponse, FinancialMetrics
from app.services.vector_store import VectorStore

EXTRACTION_PROMPT = """Extract financial metrics from the following document text.
Return a JSON object matching this exact schema. Use null for any field not found in the text.
Do NOT invent or estimate values — only extract what is explicitly stated.

Schema:
{
  "company_name": string | null,
  "fiscal_year": string | null,
  "filing_type": "10-K" | "10-Q" | "earnings_release" | null,
  "metrics": {
    "revenue": {"value": number | null, "unit": "USD_billions" | "USD_millions" | null, "period": string | null} | null,
    "eps": {"value": number | null, "diluted": boolean | null} | null,
    "net_income": {"value": number | null, "unit": string | null} | null,
    "gross_margin": {"value": number | null} | null,
    "guidance": {"revenue_low": number | null, "revenue_high": number | null, "period": string | null} | null,
    "yoy_deltas": {"revenue": number | null, "eps": number | null, "net_income": number | null} | null
  }
}

Document text:
{document_text}

Return ONLY the JSON object, no other text."""


class Extractor:
    def __init__(
        self,
        vector_store: VectorStore,
        anthropic_client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._store = vector_store
        self._client = anthropic_client
        self._model = model

    def extract(self, document_id: str) -> ExtractionResponse:
        # Reassemble document text from stored chunks in order
        raw_chunks = self._store.get_all_chunks(document_id)
        sorted_chunks = sorted(raw_chunks, key=lambda c: c["metadata"].get("chunk_index", 0))
        document_text = "\n\n".join(c["text"] for c in sorted_chunks)

        prompt = EXTRACTION_PROMPT.format(document_text=document_text[:50_000])  # cap at 50k chars

        message = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = message.content[0].text

        try:
            data = json.loads(raw_text)
            return ExtractionResponse(
                document_id=document_id,
                company_name=data.get("company_name"),
                fiscal_year=data.get("fiscal_year"),
                filing_type=data.get("filing_type"),
                metrics=FinancialMetrics(**data.get("metrics", {})),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            # Claude returned non-JSON — return an empty extraction
            return ExtractionResponse(document_id=document_id)
```

**Step 4: Run — verify PASS**

```bash
pytest tests/test_extractor.py -v
# Expected: 3 passed
```

**Step 5: Commit**

```bash
git add backend/app/services/extractor.py backend/tests/test_extractor.py
git commit -m "feat: add structured financial metric extractor with JSON schema prompting"
```

---

### Task 9: FastAPI app + dependency injection

**Branch:** `feat/fastapi-app`

**Why this matters:** `Depends()` is FastAPI's dependency injection system. It's how we share expensive objects (Embedder loads a model, VectorStore connects to ChromaDB) across requests without recreating them. This is the FastAPI equivalent of a singleton service in other frameworks.

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/dependencies.py`

**Step 1: Write `backend/app/dependencies.py`**

```python
# dependencies.py — singleton service instances via FastAPI Depends()
#
# Each function decorated with @lru_cache is called once per process.
# FastAPI's Depends() calls these to resolve constructor parameters.
# Tests override these with mock versions using app.dependency_overrides.

from functools import lru_cache

import anthropic

from app.config import get_settings
from app.services.embedder import Embedder
from app.services.extractor import Extractor
from app.services.rag_pipeline import RagPipeline
from app.services.vector_store import VectorStore


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(model_name=settings.embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(persist_path=settings.chroma_persist_path)


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    settings = get_settings()
    return RagPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        anthropic_client=get_anthropic_client(),
        model=settings.claude_model,
        top_k=settings.top_k_chunks,
    )


@lru_cache
def get_extractor() -> Extractor:
    settings = get_settings()
    return Extractor(
        vector_store=get_vector_store(),
        anthropic_client=get_anthropic_client(),
        model=settings.claude_model,
    )
```

**Step 2: Write `backend/app/main.py`**

```python
# main.py — FastAPI application entry point
#
# CORS (Cross-Origin Resource Sharing):
# The browser blocks API calls from one origin (localhost:3000) to another
# (localhost:8000) by default. We add CORS middleware to allow the frontend
# to call the backend. In production, replace "*" with the Railway frontend URL.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import extract, ingest, query

app = FastAPI(
    title="Financial Document Intelligence API",
    description="RAG-powered Q&A for financial documents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tightened to specific URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["ingestion"])
app.include_router(query.router, tags=["query"])
app.include_router(extract.router, tags=["extraction"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

**Step 3: Commit**

```bash
git add backend/app/main.py backend/app/dependencies.py
git commit -m "feat: add FastAPI app with CORS middleware and dependency injection"
```

---

### Task 10: /ingest endpoint

**Branch:** `feat/ingest-endpoint`

**Files:**
- Create: `backend/app/api/ingest.py`
- Extend: `backend/tests/test_api.py`

**Step 1: Write failing test**

```python
# backend/tests/test_api.py
import io
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_embedder, get_vector_store
from app.models.document import Chunk, DocumentStatus


@pytest.fixture
def mock_embedder():
    m = MagicMock()
    m.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    return m


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def client(mock_embedder, mock_vector_store):
    # Override real services with mocks for the duration of this test
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ingest_returns_document_id(client):
    with patch("app.api.ingest.PdfProcessor") as MockProcessor:
        mock_proc = MagicMock()
        mock_proc.process.return_value = [
            Chunk(
                id="c1", document_id="doc-1", section_title="MD&A",
                section_type="mda", text="Revenue...", page_number=1, chunk_index=0
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
```

**Step 2: Run — verify FAIL**

```bash
pytest tests/test_api.py -v
```

**Step 3: Write `backend/app/api/ingest.py`**

```python
# ingest.py — POST /ingest endpoint
#
# UploadFile is FastAPI's async file wrapper.
# We write the PDF to a temp file because pdfplumber needs a file path
# (it uses the underlying pdfminer which requires seekable file access).
# tempfile.NamedTemporaryFile with delete=False lets us control cleanup.

import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.dependencies import get_embedder, get_vector_store
from app.models.document import DocumentStatus, IngestResponse
from app.services.embedder import Embedder
from app.services.pdf_processor import PdfProcessor
from app.services.vector_store import VectorStore
from app.config import get_settings

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    settings = get_settings()
    document_id = str(uuid.uuid4())

    # Write uploaded bytes to a temp file for pdfplumber
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        contents = await file.read()
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

        # Embed all chunk texts in one batch (faster than individual calls)
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
        os.unlink(tmp_path)  # Always clean up the temp file
```

**Step 4: Run — verify PASS**

```bash
pytest tests/test_api.py -v
# Expected: 2 passed
```

**Step 5: Commit**

```bash
git add backend/app/api/ingest.py backend/tests/test_api.py
git commit -m "feat: add POST /ingest endpoint with PDF validation and batch embedding"
```

---

### Task 11: /query and /extract endpoints

**Branch:** `feat/query-extract-endpoints`

**Files:**
- Create: `backend/app/api/query.py`
- Create: `backend/app/api/extract.py`
- Extend: `backend/tests/test_api.py`

**Step 1: Add tests**

```python
# Add to backend/tests/test_api.py

from app.dependencies import get_rag_pipeline, get_extractor
from app.models.financial import (
    Citation, ExtractionResponse, FinancialMetrics,
    QueryResponse, RevenueMetric
)


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
def full_client(mock_embedder, mock_vector_store, mock_rag_pipeline, mock_extractor):
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag_pipeline
    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_returns_answer_and_citations(full_client):
    response = full_client.post(
        "/query",
        json={"document_id": "doc-1", "question": "What drove revenue?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["citations"]) == 1
    assert data["citations"][0]["section_title"] == "MD&A"


def test_extract_returns_metrics(full_client):
    response = full_client.post("/extract", json={"document_id": "doc-1"})
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Apple Inc."
    assert data["metrics"]["revenue"]["value"] == 391.0
```

**Step 2: Write `backend/app/api/query.py`**

```python
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
```

**Step 3: Write `backend/app/api/extract.py`**

```python
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
```

**Step 4: Run all backend tests**

```bash
pytest tests/ -v
# Expected: all pass, coverage >80%
```

**Step 5: Run linting**

```bash
ruff check app/
mypy app/
```

**Step 6: Commit**

```bash
git add backend/app/api/query.py backend/app/api/extract.py backend/tests/test_api.py
git commit -m "feat: add POST /query and POST /extract endpoints"
```

---

### Task 12: Dockerfile + docker-compose

**Branch:** `chore/docker`

**Why this matters:** Multi-stage Docker builds keep the final image small. Stage 1 (`builder`) installs all deps. Stage 2 (`runtime`) copies only the installed packages — no build tools, no pip cache. This is a standard production pattern.

**Files:**
- Create: `backend/Dockerfile`
- Create: `docker-compose.yml`

**Step 1: Write `backend/Dockerfile`**

```dockerfile
# Stage 1: install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "."

# Stage 2: runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Write `docker-compose.yml`**

```yaml
# docker-compose.yml — local development
#
# The 'backend' service mounts ./backend as a volume so code changes
# are reflected immediately without rebuilding the image.
# In CI/prod we build the image without the volume mount.

version: "3.9"

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - CHROMA_PERSIST_PATH=/data/chroma_db
    volumes:
      - ./backend/app:/app/app          # hot reload in dev
      - chroma_data:/data/chroma_db     # persistent vector store
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

volumes:
  chroma_data:
```

**Step 3: Commit**

```bash
git add backend/Dockerfile docker-compose.yml
git commit -m "chore: add multi-stage Dockerfile and docker-compose for local dev"
```

---

## Phase 2 — CI/CD

---

### Task 13: GitHub Actions CI

**Branch:** `chore/github-actions`

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `sonar-project.properties`

**Step 1: Write `.github/workflows/ci.yml`**

```yaml
# ci.yml — runs on every push and pull request
#
# We use a matrix to run tests on Python 3.12 only (YAGNI — no need for
# multi-version testing on a portfolio project).
#
# The Anthropic API key is stored as a GitHub Actions secret.
# In tests, it's mocked, so we pass a fake value to satisfy pydantic-settings.

name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check app/

      - name: Type check with mypy
        run: mypy app/

      - name: Run tests with coverage
        env:
          ANTHROPIC_API_KEY: "test-key-for-ci"
        run: pytest tests/ --cov=app --cov-report=xml --cov-fail-under=80

      - name: Upload coverage to SonarCloud
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: backend/coverage.xml

  sonarcloud:
    needs: backend
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # SonarCloud needs full git history

      - uses: actions/download-artifact@v4
        with:
          name: coverage-report
          path: backend/

      - uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

**Step 2: Write `sonar-project.properties`**

```properties
sonar.projectKey=financial-document-intelligence
sonar.organization=your-org-name
sonar.sources=backend/app
sonar.tests=backend/tests
sonar.python.coverage.reportPaths=backend/coverage.xml
sonar.python.version=3.12
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml sonar-project.properties
git commit -m "chore: add GitHub Actions CI with ruff, mypy, pytest coverage, and SonarCloud"
```

---

## Phase 3 — Frontend

---

### Task 14: Next.js scaffold with shadcn/ui

**Branch:** `feat/frontend-scaffold`

**Why this matters:** The App Router (introduced in Next.js 13) uses React Server Components by default. Components marked `"use client"` opt in to browser-side interactivity. For our app, most components need client-side state (file upload, query input, results) so they'll be client components. The layout is a server component.

**Step 1: Scaffold Next.js**

```bash
cd /path/to/project
npx create-next-app@latest frontend \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir \
  --no-import-alias
```

**Step 2: Install shadcn/ui**

```bash
cd frontend
npx shadcn@latest init
# Choose: Default style, Slate base color, CSS variables: yes
```

**Step 3: Install shadcn components we'll use**

```bash
npx shadcn@latest add button card badge separator skeleton scroll-area input textarea
```

**Step 4: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold Next.js 14 App Router with shadcn/ui and Tailwind"
```

---

### Task 15: TypeScript types + API client

**Branch:** `feat/frontend-api-client`

**Why this matters:** Duplicating types between backend and frontend is unavoidable in a non-monorepo setup. We mirror the Pydantic models as TypeScript interfaces. The API client centralises all `fetch` calls so components never hardcode URLs or worry about error handling.

**Files:**
- Create: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/api.ts`

**Step 1: Write `frontend/src/lib/types.ts`**

```typescript
// types.ts — mirrors backend Pydantic models
// Keep in sync with backend/app/models/

export type DocumentStatus = "processing" | "ready" | "error";

export interface IngestResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  chunks_count: number;
  sections_found: string[];
}

export interface Citation {
  section_title: string;
  excerpt: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
}

export interface RevenueMetric {
  value: number | null;
  unit: string | null;
  period: string | null;
}

export interface EpsMetric {
  value: number | null;
  diluted: boolean | null;
}

export interface FinancialMetrics {
  revenue: RevenueMetric | null;
  eps: EpsMetric | null;
  net_income: { value: number | null; unit: string | null } | null;
  gross_margin: { value: number | null } | null;
  guidance: {
    revenue_low: number | null;
    revenue_high: number | null;
    period: string | null;
  } | null;
  yoy_deltas: {
    revenue: number | null;
    eps: number | null;
    net_income: number | null;
  } | null;
}

export interface ExtractionResponse {
  document_id: string;
  company_name: string | null;
  fiscal_year: string | null;
  filing_type: string | null;
  metrics: FinancialMetrics;
}

// Client-side document representation (augments IngestResponse)
export interface Document extends IngestResponse {
  extraction?: ExtractionResponse;
}
```

**Step 2: Write `frontend/src/lib/api.ts`**

```typescript
// api.ts — typed API client
//
// All fetch calls go through these functions.
// NEXT_PUBLIC_API_URL is set at build time via environment variable.
// In local dev: http://localhost:8000
// In Railway production: https://your-backend.railway.app

import type {
  ExtractionResponse,
  IngestResponse,
  QueryResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text);
  }
  return res.json() as Promise<T>;
}

export async function ingestDocument(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: form });
  return handleResponse<IngestResponse>(res);
}

export async function queryDocument(
  documentId: string,
  question: string
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, question }),
  });
  return handleResponse<QueryResponse>(res);
}

export async function extractMetrics(
  documentId: string
): Promise<ExtractionResponse> {
  const res = await fetch(`${API_BASE}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId }),
  });
  return handleResponse<ExtractionResponse>(res);
}
```

**Step 3: Commit**

```bash
git add frontend/src/lib/
git commit -m "feat: add TypeScript types and typed API client"
```

---

### Task 16: DocumentSidebar + UploadZone components

**Branch:** `feat/frontend-sidebar-upload`

**Files:**
- Create: `frontend/src/components/UploadZone.tsx`
- Create: `frontend/src/components/DocumentSidebar.tsx`
- Create: `frontend/src/hooks/useDocuments.ts`

**Step 1: Write `frontend/src/hooks/useDocuments.ts`**

```typescript
// useDocuments.ts — manages the document list state
// This is a custom React hook. Hooks let us extract stateful logic from
// components so it can be shared and tested independently.

"use client";

import { useState, useCallback } from "react";
import { ingestDocument, extractMetrics } from "@/lib/api";
import type { Document } from "@/lib/types";

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const ingestResult = await ingestDocument(file);
      const extraction = await extractMetrics(ingestResult.document_id);
      const doc: Document = { ...ingestResult, extraction };
      setDocuments((prev) => [doc, ...prev]);
      setSelectedId(doc.document_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  const selectedDocument = documents.find((d) => d.document_id === selectedId);

  return { documents, selectedDocument, selectedId, setSelectedId, upload, uploading, error };
}
```

**Step 2: Write `frontend/src/components/UploadZone.tsx`**

```tsx
// UploadZone.tsx — drag-and-drop PDF upload
// The dragover/dragenter/dragleave/drop events handle the visual feedback.
// We prevent default on dragover to allow drop (browser default blocks it).

"use client";

import { useCallback, useState } from "react";
import { Upload, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onUpload: (file: File) => void;
  uploading: boolean;
}

export function UploadZone({ onUpload, uploading }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file?.type === "application/pdf") onUpload(file);
    },
    [onUpload]
  );

  return (
    <label
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 cursor-pointer transition-colors",
        dragging
          ? "border-blue-500 bg-blue-50"
          : "border-slate-300 hover:border-slate-400 hover:bg-slate-50"
      )}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
        }}
        disabled={uploading}
      />
      {uploading ? (
        <>
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
          <p className="text-sm text-slate-600">Processing document…</p>
        </>
      ) : (
        <>
          <Upload className="h-8 w-8 text-slate-400" />
          <div className="text-center">
            <p className="text-sm font-medium text-slate-700">Drop a PDF here</p>
            <p className="text-xs text-slate-500 mt-1">or click to browse</p>
          </div>
        </>
      )}
    </label>
  );
}
```

**Step 3: Write `frontend/src/components/DocumentSidebar.tsx`**

```tsx
"use client";

import { FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { UploadZone } from "./UploadZone";
import type { Document } from "@/lib/types";

interface DocumentSidebarProps {
  documents: Document[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  uploading: boolean;
}

export function DocumentSidebar({
  documents,
  selectedId,
  onSelect,
  onUpload,
  uploading,
}: DocumentSidebarProps) {
  return (
    <aside className="flex h-full w-72 flex-shrink-0 flex-col border-r bg-slate-50">
      <div className="p-4">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
          Documents
        </h2>
        <UploadZone onUpload={onUpload} uploading={uploading} />
      </div>

      <Separator />

      <ScrollArea className="flex-1 p-2">
        {documents.length === 0 ? (
          <p className="p-4 text-center text-xs text-slate-400">
            No documents yet
          </p>
        ) : (
          <ul className="space-y-1">
            {documents.map((doc) => (
              <li key={doc.document_id}>
                <button
                  onClick={() => onSelect(doc.document_id)}
                  className={cn(
                    "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
                    selectedId === doc.document_id
                      ? "bg-blue-100 text-blue-900"
                      : "hover:bg-slate-100 text-slate-700"
                  )}
                >
                  <div className="flex items-start gap-2">
                    <FileText className="mt-0.5 h-4 w-4 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">
                        {doc.extraction?.company_name ?? doc.filename}
                      </p>
                      <div className="mt-1 flex items-center gap-1.5">
                        {doc.extraction?.fiscal_year && (
                          <Badge variant="secondary" className="text-xs">
                            {doc.extraction.fiscal_year}
                          </Badge>
                        )}
                        {doc.extraction?.filing_type && (
                          <Badge variant="outline" className="text-xs">
                            {doc.extraction.filing_type}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </ScrollArea>
    </aside>
  );
}
```

**Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: add DocumentSidebar, UploadZone, and useDocuments hook"
```

---

### Task 17: MetricsDashboard component

**Branch:** `feat/frontend-metrics-dashboard`

**Files:**
- Create: `frontend/src/components/MetricsDashboard.tsx`

**Step 1: Write `frontend/src/components/MetricsDashboard.tsx`**

```tsx
// MetricsDashboard.tsx — displays structured extracted metrics
// Each metric card shows the value with its unit and a YoY delta badge.
// The delta is green for positive, red for negative — standard finance UI.

"use client";

import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ExtractionResponse } from "@/lib/types";

interface MetricCardProps {
  title: string;
  value: string | null;
  delta?: number | null;
  subtitle?: string;
}

function MetricCard({ title, value, delta, subtitle }: MetricCardProps) {
  const DeltaIcon =
    delta == null ? null : delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus;
  const deltaColor =
    delta == null ? "" : delta > 0 ? "text-emerald-600" : delta < 0 ? "text-red-500" : "text-slate-500";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-slate-500">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-bold text-slate-900">
          {value ?? <span className="text-slate-300">—</span>}
        </p>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        {delta != null && DeltaIcon && (
          <div className={`flex items-center gap-1 mt-1.5 text-sm font-medium ${deltaColor}`}>
            <DeltaIcon className="h-4 w-4" />
            <span>{delta > 0 ? "+" : ""}{delta.toFixed(1)}% YoY</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface MetricsDashboardProps {
  extraction: ExtractionResponse;
}

function formatBillions(value: number | null | undefined, unit?: string | null): string | null {
  if (value == null) return null;
  const label = unit === "USD_billions" ? "B" : unit === "USD_millions" ? "M" : "";
  return `$${value.toFixed(1)}${label}`;
}

export function MetricsDashboard({ extraction }: MetricsDashboardProps) {
  const { metrics, company_name, fiscal_year, filing_type } = extraction;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-semibold text-slate-900">
          {company_name ?? "Company"} — {fiscal_year ?? "Unknown Year"}
        </h3>
        {filing_type && (
          <Badge variant="outline">{filing_type}</Badge>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricCard
          title="Revenue"
          value={formatBillions(metrics.revenue?.value, metrics.revenue?.unit)}
          subtitle={metrics.revenue?.period ?? undefined}
          delta={metrics.yoy_deltas?.revenue}
        />
        <MetricCard
          title="EPS"
          value={metrics.eps?.value != null ? `$${metrics.eps.value.toFixed(2)}` : null}
          subtitle={metrics.eps?.diluted ? "diluted" : undefined}
          delta={metrics.yoy_deltas?.eps}
        />
        <MetricCard
          title="Net Income"
          value={formatBillions(metrics.net_income?.value, metrics.net_income?.unit)}
          delta={metrics.yoy_deltas?.net_income}
        />
        <MetricCard
          title="Gross Margin"
          value={metrics.gross_margin?.value != null ? `${metrics.gross_margin.value.toFixed(1)}%` : null}
        />
        {metrics.guidance && (
          <MetricCard
            title="Guidance"
            value={
              metrics.guidance.revenue_low != null
                ? `$${metrics.guidance.revenue_low}–${metrics.guidance.revenue_high}B`
                : null
            }
            subtitle={metrics.guidance.period ?? undefined}
          />
        )}
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/MetricsDashboard.tsx
git commit -m "feat: add MetricsDashboard with YoY delta indicators"
```

---

### Task 18: QueryInterface + CitationCard components

**Branch:** `feat/frontend-query-ui`

**Files:**
- Create: `frontend/src/components/CitationCard.tsx`
- Create: `frontend/src/components/QueryInterface.tsx`

**Step 1: Write `frontend/src/components/CitationCard.tsx`**

```tsx
// CitationCard.tsx — displays a single retrieved passage
// The relevance score is shown as a percentage with a colour bar.
// High score (>80%) = green, medium = yellow, low = slate.

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Citation } from "@/lib/types";

interface CitationCardProps {
  citation: Citation;
  index: number;
}

function scoreColor(score: number) {
  if (score >= 0.8) return "text-emerald-700 bg-emerald-50 border-emerald-200";
  if (score >= 0.6) return "text-amber-700 bg-amber-50 border-amber-200";
  return "text-slate-600 bg-slate-50 border-slate-200";
}

export function CitationCard({ citation, index }: CitationCardProps) {
  return (
    <Card className={`border ${scoreColor(citation.score)}`}>
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-700 text-white text-xs font-bold flex-shrink-0">
              {index}
            </span>
            <span className="text-xs font-semibold">{citation.section_title}</span>
          </div>
          <Badge variant="outline" className="text-xs flex-shrink-0">
            {(citation.score * 100).toFixed(0)}% match
          </Badge>
        </div>
        <p className="text-sm leading-relaxed line-clamp-3">{citation.excerpt}</p>
      </CardContent>
    </Card>
  );
}
```

**Step 2: Write `frontend/src/components/QueryInterface.tsx`**

```tsx
"use client";

import { useState } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import { CitationCard } from "./CitationCard";
import { queryDocument } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";

interface QueryInterfaceProps {
  documentId: string;
}

export function QueryInterface({ documentId }: QueryInterfaceProps) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await queryDocument(documentId, question);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <Textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask anything about this document… e.g. 'What were the main revenue drivers?'"
          className="min-h-[60px] resize-none"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <Button type="submit" disabled={loading || !question.trim()} className="self-end">
          <Send className="h-4 w-4" />
        </Button>
      </form>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
      )}

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      )}

      {result && !loading && (
        <div className="space-y-4">
          <div className="rounded-xl bg-slate-900 text-slate-100 p-4 text-sm leading-relaxed">
            {result.answer}
          </div>

          {result.citations.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                Sources
              </h4>
              <div className="space-y-2">
                {result.citations.map((citation, i) => (
                  <CitationCard key={i} citation={citation} index={i + 1} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: add QueryInterface with loading skeletons and CitationCard"
```

---

### Task 19: Main page composition

**Branch:** `feat/frontend-main-page`

**Files:**
- Modify: `frontend/src/app/page.tsx`
- Modify: `frontend/src/app/layout.tsx`

**Step 1: Write `frontend/src/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Financial Document Intelligence",
  description: "RAG-powered Q&A for financial documents",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-white text-slate-900`}>
        {children}
      </body>
    </html>
  );
}
```

**Step 2: Write `frontend/src/app/page.tsx`**

```tsx
// page.tsx — main application layout
// Three-panel layout: sidebar | main content area
// The main area shows MetricsDashboard (top) + QueryInterface (bottom)
// when a document is selected, and an empty state otherwise.

"use client";

import { Separator } from "@/components/ui/separator";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import { QueryInterface } from "@/components/QueryInterface";
import { useDocuments } from "@/hooks/useDocuments";
import { FileSearch } from "lucide-react";

export default function Home() {
  const { documents, selectedDocument, selectedId, setSelectedId, upload, uploading, error } =
    useDocuments();

  return (
    <div className="flex h-screen overflow-hidden">
      <DocumentSidebar
        documents={documents}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onUpload={upload}
        uploading={uploading}
      />

      <main className="flex flex-1 flex-col overflow-hidden">
        {error && (
          <div className="bg-red-50 px-6 py-2 text-sm text-red-700 border-b border-red-200">
            {error}
          </div>
        )}

        {!selectedDocument ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-slate-400">
            <FileSearch className="h-16 w-16" />
            <div className="text-center">
              <p className="text-lg font-medium">No document selected</p>
              <p className="text-sm mt-1">Upload a 10-K or earnings report to get started</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 flex-col overflow-y-auto">
            {selectedDocument.extraction && (
              <>
                <div className="px-8 py-6">
                  <MetricsDashboard extraction={selectedDocument.extraction} />
                </div>
                <Separator />
              </>
            )}

            <div className="flex-1 px-8 py-6">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500 mb-4">
                Ask a Question
              </h3>
              <QueryInterface documentId={selectedDocument.document_id} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/app/
git commit -m "feat: compose main page with three-panel layout"
```

---

### Task 20: Frontend Dockerfile

**Branch:** `chore/frontend-docker`

**Why this matters:** Next.js standalone output mode creates a minimal production build that doesn't need node_modules at runtime. The image goes from ~1GB to ~150MB.

**Files:**
- Create: `frontend/Dockerfile`
- Modify: `frontend/next.config.ts`

**Step 1: Update `frontend/next.config.ts`**

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",  // enables minimal Docker image
};

export default nextConfig;
```

**Step 2: Write `frontend/Dockerfile`**

```dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

**Step 3: Commit**

```bash
git add frontend/Dockerfile frontend/next.config.ts
git commit -m "chore: add multi-stage Next.js Dockerfile with standalone output"
```

---

## Phase 4 — Deployment & Docs

---

### Task 21: README

**Branch:** `docs/readme`

**Files:**
- Create: `README.md`

**Step 1: Write README**

Include:
- Project title + one-line description
- Architecture diagram (copy from design doc)
- Tech stack table
- Quick start (docker-compose up)
- API reference (endpoints, example curl commands)
- Deployment instructions (Railway)
- Link to live demo URL (fill in after deploy)
- Screenshots section (fill in after deploy)

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with architecture diagram and quick start"
```

---

### Task 22: Railway deployment

**Branch:** `chore/railway-deploy`

**Step 1: Create `railway.toml` for backend**

```toml
# backend/railway.toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
```

**Step 2: Create `railway.toml` for frontend**

```toml
# frontend/railway.toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "node server.js"
healthcheckPath = "/"
```

**Step 3: Set Railway environment variables**

In Railway dashboard:
- Backend service: `ANTHROPIC_API_KEY`, `CHROMA_PERSIST_PATH=/data/chroma`
- Frontend service: `NEXT_PUBLIC_API_URL=https://<backend>.railway.app`

**Step 4: Commit**

```bash
git add backend/railway.toml frontend/railway.toml
git commit -m "chore: add Railway deployment configuration"
```

---

## Verification Checklist

Before calling the project done, verify:

- [ ] `pytest tests/ --cov=app --cov-fail-under=80` passes locally
- [ ] `ruff check app/` reports no errors
- [ ] `mypy app/` reports no errors
- [ ] `docker-compose up` starts both services without errors
- [ ] Upload a real 10-K PDF → sidebar shows company name + year
- [ ] Metrics dashboard shows revenue, EPS, etc.
- [ ] Ask a question → answer appears with citation cards
- [ ] GitHub Actions CI passes on push
- [ ] SonarCloud quality gate is green
- [ ] Live Railway URL is accessible

---

## Learning Checkpoints

At each phase, pause and understand:

1. **After Task 4 (PDF Processor):** Why is section-aware chunking better for retrieval than fixed-size chunking? What would a question about "risk factors" retrieve with each approach?

2. **After Task 7 (RAG Pipeline):** Read the `SYSTEM_PROMPT` carefully. Why do we number context blocks? What happens if we don't include the instruction to cite by number?

3. **After Task 8 (Extractor):** Compare the extractor prompt to the RAG prompt. Why is the extractor prompt more prescriptive? Why do we explicitly include the JSON schema?

4. **After Task 13 (CI):** Look at the GitHub Actions YAML. Why does `sonarcloud` job have `needs: backend`? What does `fetch-depth: 0` do?

5. **After Task 17 (MetricsDashboard):** Why do all metric fields use `null` instead of `0` for missing values? What would a recruiter see if we used `0`?
