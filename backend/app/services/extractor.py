import json
from typing import Any

import anthropic
from anthropic.types import TextBlock

from app.models.financial import ExtractionResponse, FinancialMetrics
from app.services.vector_store import VectorStore

EXTRACTION_PROMPT = """Extract financial metrics from the following document text.
Return a JSON object matching this exact schema. Use null for any field not found in the text.
Do NOT invent or estimate values — only extract what is explicitly stated.

Schema:
{{
  "company_name": string | null,
  "fiscal_year": string | null,
  "filing_type": "10-K" | "10-Q" | "earnings_release" | null,
  "metrics": {{
    "revenue": {{"value": number | null, "unit": "USD_billions" | "USD_millions" | null, "period": string | null}} | null,
    "eps": {{"value": number | null, "diluted": boolean | null}} | null,
    "net_income": {{"value": number | null, "unit": string | null}} | null,
    "gross_margin": {{"value": number | null}} | null,
    "guidance": {{"revenue_low": number | null, "revenue_high": number | null, "period": string | null}} | null,
    "yoy_deltas": {{"revenue": number | null, "eps": number | null, "net_income": number | null}} | null
  }}
}}

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
        raw_chunks = self._store.get_all_chunks(document_id)
        sorted_chunks = sorted(raw_chunks, key=lambda c: c["metadata"].get("chunk_index", 0))

        # Financial data lives in Item 7 (MD&A) and Item 8 (Financial Statements).
        # Prioritise those sections so they are never cut off by the character limit.
        FINANCIAL_KEYWORDS = ("item 7", "item 8", "md&a", "management", "financial statement",
                              "results of operations", "revenue", "income", "earnings")

        def is_financial(chunk: dict[str, Any]) -> bool:
            title = str(chunk["metadata"].get("section_title", "")).lower()
            return any(kw in title for kw in FINANCIAL_KEYWORDS)

        financial_chunks = [c for c in sorted_chunks if is_financial(c)]
        other_chunks = [c for c in sorted_chunks if not is_financial(c)]
        ordered = financial_chunks + other_chunks

        document_text = "\n\n".join(c["text"] for c in ordered)
        prompt = EXTRACTION_PROMPT.format(document_text=document_text[:150_000])

        message = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        block = message.content[0]
        raw_text = block.text if isinstance(block, TextBlock) else getattr(block, "text", "")

        try:
            data = json.loads(raw_text)
            return ExtractionResponse(
                document_id=document_id,
                company_name=data.get("company_name"),
                fiscal_year=data.get("fiscal_year"),
                filing_type=data.get("filing_type"),
                metrics=FinancialMetrics(**data.get("metrics", {})),
            )
        except (TypeError, ValueError):
            return ExtractionResponse(document_id=document_id)
