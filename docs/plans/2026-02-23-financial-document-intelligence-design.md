# Financial Document Intelligence Agent — Design Doc
_Date: 2026-02-23_

## Purpose

A portfolio-grade RAG-powered Q&A system for financial documents (earnings
reports, 10-K filings). Primary audience: recruiters and hiring managers for
platform/AI engineering roles.

**The demo story:** Upload a 10-K → AI extracts key metrics as structured JSON
→ ask any natural language question → get a grounded answer with the exact
source paragraphs highlighted.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary showpiece | `/query` with citations | Demonstrates full RAG architecture and hallucination mitigation |
| Chunking strategy | Section-aware | Detects 10-K section headers (Item 1, MD&A, Risk Factors), preserves semantic boundaries |
| Frontend quality | Polished (shadcn/ui) | Owner is a platform frontend engineer — the UI should reflect that |
| Deployment | Two Railway services | FastAPI + Next.js as separate services; realistic microservices architecture |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                         │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────┐   │
│  │ Doc Sidebar  │  │  Query + Chat  │  │ Metrics Dashboard │   │
│  │ (upload/list)│  │ (citations UI) │  │ (extracted JSON)  │   │
│  └──────────────┘  └────────────────┘  └───────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST / JSON
┌────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Backend                           │
│  POST /ingest          POST /query           POST /extract      │
│  ┌──────────────┐  ┌────────────────────┐  ┌────────────────┐  │
│  │PDF Processor │  │   RAG Pipeline     │  │  Extractor     │  │
│  │- pdfplumber  │  │ 1. embed question  │  │ Claude +       │  │
│  │- section     │  │ 2. ChromaDB search │  │ Pydantic JSON  │  │
│  │  detection   │  │ 3. build prompt    │  │ schema         │  │
│  │- chunking    │  │ 4. Claude answer   │  └────────────────┘  │
│  └──────┬───────┘  │ 5. return citations│                       │
│         │          └────────┬───────────┘                       │
│  ┌──────▼───────┐           │                                   │
│  │  Embedder    │    ┌──────▼───────┐    ┌──────────────────┐  │
│  │ (sentence-   │    │   ChromaDB   │    │  Anthropic SDK   │  │
│  │  transformers│    │ (vector store│    │  claude-sonnet-  │  │
│  │  MiniLM-L6)  │    │  + metadata) │    │  4-6             │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Contracts

### `POST /ingest`
```json
// Request: multipart/form-data { file: PDF }
// Response:
{
  "document_id": "uuid",
  "filename": "apple-10k-2024.pdf",
  "status": "ready",
  "chunks_count": 47,
  "sections_found": ["Business Overview", "Risk Factors", "MD&A", "Financial Statements"]
}
```

### `POST /query`
```json
// Request:
{ "document_id": "uuid", "question": "What were the main revenue drivers?" }

// Response:
{
  "answer": "Apple's revenue growth was primarily driven by...",
  "citations": [
    {
      "section_title": "Management's Discussion and Analysis",
      "excerpt": "iPhone revenue increased 6% year-over-year to $205.5 billion...",
      "score": 0.91
    }
  ]
}
```

### `POST /extract`
```json
// Request:
{ "document_id": "uuid" }

// Response:
{
  "company_name": "Apple Inc.",
  "fiscal_year": "2024",
  "filing_type": "10-K",
  "metrics": {
    "revenue":      { "value": 391.0,  "unit": "USD_billions", "period": "FY2024" },
    "eps":          { "value": 6.11,   "diluted": true },
    "net_income":   { "value": 93.7,   "unit": "USD_billions" },
    "gross_margin": { "value": 46.2 },
    "guidance":     { "revenue_low": null, "revenue_high": null, "period": null },
    "yoy_deltas":   { "revenue": 2.0, "eps": 10.9, "net_income": 3.4 }
  }
}
```

---

## Data Flow

### Ingestion
```
PDF upload → pdfplumber extracts raw text → section detector (regex on
Item headers) splits into named sections → long sections sub-chunked at
~800 tokens with 100-token overlap → MiniLM-L6 embeds each chunk →
ChromaDB stores (embedding, text, metadata{document_id, section_title,
section_type, page_number, chunk_index})
```

### Query
```
Question → MiniLM-L6 embeds → ChromaDB top-5 similarity search
(filtered by document_id) → chunks inserted into Claude prompt as
numbered context blocks → Claude generates answer referencing [1],[2]
→ response parsed into answer + citations array
```

### Extract
```
Document retrieved from ChromaDB (all chunks) → assembled into ordered
text → Claude called with JSON schema prompt + Pydantic model →
structured metrics returned, null for unavailable fields
```

---

## Project Structure

```
financial-document-intelligence-agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + CORS
│   │   ├── config.py            # pydantic-settings (ANTHROPIC_API_KEY, etc.)
│   │   ├── api/
│   │   │   ├── ingest.py        # POST /ingest
│   │   │   ├── query.py         # POST /query
│   │   │   └── extract.py       # POST /extract
│   │   ├── services/
│   │   │   ├── pdf_processor.py  # pdfplumber + section detection
│   │   │   ├── embedder.py       # sentence-transformers wrapper
│   │   │   ├── vector_store.py   # ChromaDB wrapper
│   │   │   ├── rag_pipeline.py   # retrieval + prompt + Claude
│   │   │   └── extractor.py      # structured extraction via Claude
│   │   └── models/
│   │       ├── document.py       # Document, Chunk, IngestResponse
│   │       └── financial.py      # QueryResponse, ExtractionResponse
│   ├── tests/
│   │   ├── conftest.py           # fixtures, mocks
│   │   ├── test_api.py           # endpoint integration tests
│   │   ├── test_pdf_processor.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_extractor.py
│   ├── Dockerfile
│   └── pyproject.toml            # deps + ruff + mypy config
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── ui/               # shadcn/ui primitives
│   │   │   ├── DocumentSidebar.tsx
│   │   │   ├── UploadZone.tsx
│   │   │   ├── QueryInterface.tsx
│   │   │   ├── CitationCard.tsx
│   │   │   └── MetricsDashboard.tsx
│   │   ├── lib/
│   │   │   ├── api.ts            # typed fetch client
│   │   │   └── types.ts          # mirrors backend Pydantic models
│   │   └── hooks/
│   │       └── useDocuments.ts
│   └── Dockerfile
├── .github/
│   └── workflows/
│       ├── ci.yml                # ruff + mypy + pytest --cov
│       └── sonar.yml             # SonarCloud scan
├── docker-compose.yml            # local dev: backend + frontend
├── sonar-project.properties
└── README.md
```

---

## Technology Choices

| Layer | Library | Why |
|---|---|---|
| PDF extraction | pdfplumber | Better table/layout handling than PyPDF2; actively maintained |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Fast, small (80MB), strong semantic quality; no API cost |
| Vector store | ChromaDB | In-process (no separate service), persistent, good Python API |
| LLM | Anthropic `claude-sonnet-4-6` | Specified in brief; best balance of quality/speed/cost |
| Backend | FastAPI + pydantic-settings | Async, typed, automatic OpenAPI docs |
| Frontend | Next.js 14 App Router + shadcn/ui | Modern patterns; Tailwind-based, accessible components |
| Testing | pytest + pytest-mock + pytest-cov | Standard Python; mocks prevent real API calls in CI |
| Linting | ruff + mypy | Fast, modern Python toolchain |

---

## Testing Strategy

- All Anthropic SDK calls mocked with `pytest-mock` — CI never hits real API
- ChromaDB uses an in-memory instance in tests (no persistent state)
- `httpx.AsyncClient` for FastAPI endpoint integration tests
- Target: `>80%` line coverage enforced in CI
- SonarCloud quality gate blocks merge on coverage drop or new bugs

---

## Deployment

```
Railway Project
├── Service: backend   (Docker, FastAPI on port 8000)
│   └── Env: ANTHROPIC_API_KEY, CHROMA_PERSIST_PATH
└── Service: frontend  (Docker, Next.js on port 3000)
    └── Env: NEXT_PUBLIC_API_URL=<backend Railway URL>
```

Local dev: `docker-compose up` starts both services.

---

## What This Teaches (learning notes)

- **RAG architecture**: chunking → embedding → retrieval → prompt construction → cited generation
- **Section-aware chunking**: regex header detection, fallback to fixed-size splits
- **Grounded prompting**: how to structure context blocks so the model cites them reliably
- **Structured LLM output**: Pydantic schema + JSON mode → reliable extraction
- **Vector databases**: ChromaDB collection management, metadata filtering, similarity scoring
- **FastAPI patterns**: dependency injection, async file upload, Pydantic request/response models
- **Frontend**: shadcn/ui composition, citation highlight UX, loading skeleton patterns
