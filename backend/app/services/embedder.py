from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = TextEmbedding(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single string → dense vector."""
        return next(iter(self._model.embed([text]))).tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one forward pass (faster than looping)."""
        return [v.tolist() for v in self._model.embed(texts)]  # type: ignore[no-any-return]
