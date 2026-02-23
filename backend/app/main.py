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
    allow_origins=["*"],
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
