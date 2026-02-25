import anthropic
from anthropic.types import TextBlock

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


def _extract_text(block: object) -> str:
    """Extract text from a content block, handling TextBlock and duck-typed mocks."""
    if isinstance(block, TextBlock):
        return block.text
    text = getattr(block, "text", "")
    return text if isinstance(text, str) else ""


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
        question_embedding = self._embedder.embed(question)
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
        context = _build_context_block(results)
        user_message = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = _extract_text(message.content[0])
        citations = [
            Citation(section_title=r.section_title, excerpt=r.excerpt, score=r.score)
            for r in results
        ]
        return QueryResponse(answer=answer, citations=citations)
