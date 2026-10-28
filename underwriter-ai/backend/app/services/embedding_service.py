"""Dense embeddings with BGE-M3 (configurable via EMBEDDING_MODEL).

The same model embeds document chunks and user queries. Vectors are L2-normalised so
Qdrant's cosine distance equals the dot product. The model loads lazily on first use and
is shared across requests; tests inject a lightweight `Embedder` instead.
"""

import logging
import threading
from typing import Protocol

import numpy as np

from app.core.config import get_settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """BGE-M3 (or any sentence-transformers model) on CPU or GPU."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s (first run downloads it)", model_name)
        self.model = SentenceTransformer(model_name)
        # Chunks are ~300 words; capping the sequence length bounds the cost of long table chunks
        # (BGE-M3 accepts 8192 tokens, and attention cost grows quadratically with length).
        self.model.max_seq_length = min(self.model.max_seq_length or 1024, 1024)
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                            show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE-M3 needs no query instruction prefix (unlike bge-*-v1.5).
        return self.embed_documents([text])[0]


_lock = threading.Lock()
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            s = get_settings()
            _embedder = SentenceTransformerEmbedder(s.embedding_model, s.embedding_batch_size)
        return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Override the embedder (tests, alternative backends)."""
    global _embedder
    with _lock:
        _embedder = embedder
