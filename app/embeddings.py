"""Local embeddings with sentence-transformers. Nothing here leaves your machine."""

import threading

import numpy as np

from app.config import get_settings

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:  # two requests arriving at once should load the model only once
        if _model is None:
            from sentence_transformers import SentenceTransformer  # slow import, so do it lazily

            _model = SentenceTransformer(get_settings().embed_model)
    return _model


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return one L2-normalised vector per text, so dot product == cosine similarity."""
    if not texts:
        return np.zeros((0, get_settings().embed_dim), dtype=np.float32)
    vectors = _get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
