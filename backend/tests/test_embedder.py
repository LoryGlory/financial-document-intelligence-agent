from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.embedder import Embedder


@pytest.fixture
def mock_model():
    with patch("app.services.embedder.TextEmbedding") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed.return_value = iter([np.array([0.1, 0.2, 0.3])])
        mock_cls.return_value = mock_instance
        yield mock_cls


def test_embed_single_text(mock_model):
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
    result = embedder.embed("What is Apple's revenue?")
    assert isinstance(result, list)
    assert len(result) == 3


def test_embed_batch(mock_model):
    embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
    mock_model.return_value.embed.return_value = iter(
        [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
    )
    results = embedder.embed_batch(["text one", "text two"])
    assert len(results) == 2
    assert len(results[0]) == 3
