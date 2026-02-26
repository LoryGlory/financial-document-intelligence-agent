<div align="center">
  <img src="https://raw.githubusercontent.com/LoryGlory/financial-document-intelligence-agent/main/frontend/src/app/icon.svg" alt="Financial Document Intelligence logo" width="80" height="80" />
  <h1>Financial Document Intelligence Agent</h1>
  <p>by <a href="https://www.linkedin.com/in/laura-roganovic">Laura Roganovic</a></p>
</div>

[![CI](https://github.com/LoryGlory/financial-document-intelligence-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LoryGlory/financial-document-intelligence-agent/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=LoryGlory_financial-document-intelligence-agent&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=LoryGlory_financial-document-intelligence-agent)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=LoryGlory_financial-document-intelligence-agent&metric=coverage)](https://sonarcloud.io/summary/new_code?id=LoryGlory_financial-document-intelligence-agent)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)

RAG-powered Q&A for financial documents — upload a 10-K or earnings report, extract key metrics as structured JSON, then ask natural language questions and get grounded answers with inline citations.

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
│  │ (bge-small)  │    │   ChromaDB   │    │  Anthropic SDK   │  │
│  └──────────────┘    └──────────────┘    │  claude-sonnet-  │  │
│                                          │  4-6             │  │
│                                          └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

**Ingestion:** PDF upload → pdfplumber text extraction → SEC section header detection (Item 1, Item 1A, Item 7 MD&A…) → sub-chunking at ~800 tokens with 100-token overlap → fastembed ONNX embeddings → ChromaDB

**Query:** Question → embed → ChromaDB top-5 similarity search (scoped to document) → numbered context blocks → Claude generates grounded answer with `[N]` inline citations

**Extract:** Chunks sorted with financial sections first (Item 7, Item 8, Notes to Consolidated Financial Statements) → first 200k chars → Claude + JSON schema prompt → typed `ExtractionResponse` (revenue, EPS, net income, gross margin, guidance, YoY deltas)

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic `claude-sonnet-4-6` |
| Embeddings | `fastembed` ONNX (BAAI/bge-small-en-v1.5, local, no API cost) |
| Vector store | ChromaDB (persistent, in-process) |
| PDF extraction | pdfplumber |
| Backend | FastAPI + pydantic-settings |
| Frontend | Next.js 16 App Router + shadcn/ui + Tailwind CSS v4 |
| Testing | pytest + pytest-mock + pytest-cov (>80% coverage) |
| Linting | ruff + mypy (strict) |
| CI/CD | GitHub Actions + SonarCloud |
| Containerisation | Docker Compose (multi-stage builds, ONNX model baked in) |

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
echo "ANTHROPIC_API_KEY=sk-ant-..." > backend/.env

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
API_URL=http://localhost:8000 npm run dev
```

---

## API Reference

### `POST /ingest`
Upload a PDF for processing.
```bash
curl -X POST http://localhost:8000/ingest -F "file=@alphabet-10k-2025.pdf"
```
```json
{
  "document_id": "uuid",
  "filename": "alphabet-10k-2025.pdf",
  "status": "ready",
  "chunks_count": 123,
  "sections_found": ["ITEM 1. BUSINESS", "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", "ITEM 8. FINANCIAL STATEMENTS"]
}
```

### `POST /query`
Ask a natural language question about an ingested document.
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "uuid", "question": "What were the main revenue drivers in 2025?"}'
```
```json
{
  "answer": "Revenue grew 14% to $402.8B, driven by Google Search [1] and Google Cloud [2].",
  "citations": [
    { "section_title": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", "excerpt": "Google Search & other revenues increased...", "score": 0.94 }
  ]
}
```

### `POST /extract`
Extract structured financial metrics.
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"document_id": "uuid"}'
```
```json
{
  "company_name": "Alphabet Inc.",
  "fiscal_year": "2025",
  "filing_type": "10-K",
  "metrics": {
    "revenue": { "value": 402836.0, "unit": "USD_millions", "period": "Year Ended December 31, 2025" },
    "eps": { "value": 10.81, "diluted": true },
    "net_income": { "value": 132170.0, "unit": "USD_millions" },
    "gross_margin": { "value": 60.0 },
    "yoy_deltas": { "revenue": 15.0, "eps": 34.0, "net_income": 32.0 }
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
