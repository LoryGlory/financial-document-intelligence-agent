from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single string → dense vector."""
        vector = self._model.encode([text])
        return vector[0].tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one forward pass (faster than looping)."""
        vectors = self._model.encode(texts)
        return [v.tolist() for v in vectors]
