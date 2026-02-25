# Financial Document Intelligence Agent

> RAG-powered Q&A for financial documents — upload a 10-K, extract key metrics as structured JSON, then ask natural language questions and get grounded answers with inline citations.

[![CI](https://github.com/LoryGlory/financial-document-intelligence-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LoryGlory/financial-document-intelligence-agent/actions/workflows/ci.yml)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                         │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────┐   │
│  │ Doc Sidebar  │  │  Query + Chat  │  │ Metrics Dashboard │   │
│  │ (upload/list)│  │ (citations UI) │  │ (extracted JSON)  │   │
│  └──────────────┘  └────────────────┘  └───────────────────┘   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ REST / JSON
┌────────────────────────────▼─────────────────────────────────────┐
│                       FastAPI Backend                           │
│  POST /ingest          POST /query           POST /extract      │
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────┐  │
│  │PDF Processor │  │   RAG Pipeline     │  │  Extractor     │  │
│  │- pdfplumber  │  │ 1. embed question  │  │ Claude +       │  │
│  │- section     │  │ 2. ChromaDB search │  │ Pydantic JSON  │  │
│  │  detection   │  │ 3. build prompt    │  │ schema         │  │
│  │- chunking    │  │ 4. Claude answer   │  └──────────────┘  │
│  └──────┬───────┘  │ 5. return citations│                       │
│         │          └────────┬───────────┘                       │
│  ┌──────▼───────┘           │                                   │
│  │  Embedder    │    ┌──────▼───────┐    ┌──────────────────┐  │
│  │ (MiniLM-L6)  │    │   ChromaDB   │    │  Anthropic SDK   │  │
│  └──────────────┘    └──────────────┘    │  claude-sonnet-  │  │
│                                          │  4-6             │  │
│                                          └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

**Ingestion:** PDF upload → pdfplumber text extraction → SEC section header detection (Item 1, Item 1A, Item 7 MD&A…) → sub-chunking at ~800 tokens with 100-token overlap → MiniLM-L6 embeddings → ChromaDB

**Query:** Question → embed → ChromaDB top-5 similarity search (scoped to document) → numbered context blocks → Claude generates grounded answer with `[N]` inline citations

**Extract:** All chunks assembled in order → Claude + JSON schema prompt → typed `ExtractionResponse` (revenue, EPS, net income, gross margin, guidance, YoY deltas)

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic `claude-sonnet-4-6` |
| Embeddings | `sentence-transformers` MiniLM-L6-v2 (local, no API cost) |
| Vector store | ChromaDB (persistent, in-process) |
| PDF extraction | pdfplumber |
| Backend | FastAPI + pydantic-settings |
| Frontend | Next.js 16 App Router + shadcn/ui + Tailwind CSS v4 |
| Testing | pytest + pytest-mock + pytest-cov (>80% coverage) |
| Linting | ruff + mypy (strict) |
| CI/CD | GitHub Actions + SonarCloud |
| Deployment | Railway (two services) |

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- Docker + Docker Compose
- An [Anthropic API key](https://console.anthropic.com/)

### Local dev with Docker Compose

```bash
# Clone and enter the project
git clone https://github.com/LoryGlory/financial-document-intelligence-agent.git
cd financial-document-intelligence-agent

# Set your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Start both services
docker-compose up
```

Frontend: http://localhost:3000  
Backend API docs: http://localhost:8000/docs

### Local dev without Docker

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ANTHROPIC_API_KEY=sk-ant-... uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

---

## API Reference

### `POST /ingest`
Upload a PDF for processing.
```bash
curl -X POST http://localhost:8000/ingest   -F "file=@apple-10k-2024.pdf"
```
```json
{
  "document_id": "uuid",
  "filename": "apple-10k-2024.pdf",
  "status": "ready",
  "chunks_count": 47,
  "sections_found": ["Item 1. Business", "Item 1A. Risk Factors", "Item 7. MD&A"]
}
```

### `POST /query`
Ask a question about an ingested document.
```bash
curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"document_id": "uuid", "question": "What were the main revenue drivers?"}'
```
```json
{
  "answer": "Revenue grew 8% driven by iPhone sales [1] and services growth [2].",
  "citations": [
    { "section_title": "MD&A", "excerpt": "iPhone revenue increased 6%...", "score": 0.91 }
  ]
}
```

### `POST /extract`
Extract structured financial metrics.
```bash
curl -X POST http://localhost:8000/extract   -H "Content-Type: application/json"   -d '{"document_id": "uuid"}'
```
```json
{
  "company_name": "Apple Inc.",
  "fiscal_year": "2024",
  "filing_type": "10-K",
  "metrics": {
    "revenue": { "value": 391.0, "unit": "USD_billions", "period": "FY2024" },
    "eps": { "value": 6.11, "diluted": true },
    "yoy_deltas": { "revenue": 2.0, "eps": 10.9 }
  }
}
```

---

## Running Tests

```bash
cd backend
source .venv/bin/activate
pytest tests/ --cov=app --cov-report=term-missing
```

All Anthropic SDK and ChromaDB calls are mocked — tests never hit real APIs.

---

## Deployment (Railway)

1. Create a Railway project with two services: `backend` and `frontend`
2. Set environment variables:
   - Backend: `ANTHROPIC_API_KEY`, `CHROMA_PERSIST_PATH=/data/chroma`
   - Frontend: `NEXT_PUBLIC_API_URL=https://<backend>.railway.app`
3. Each service deploys from its subdirectory Dockerfile
4. Add a Railway volume on the backend service mounted at `/data/chroma` for persistent vector storage

Live demo: _coming soon_
