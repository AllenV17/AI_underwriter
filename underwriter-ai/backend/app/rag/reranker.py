"""Cross-encoder reranking of fused candidates (BGE reranker by default).

A cross-encoder reads the query and each chunk together, so it judges relevance far
better than embedding similarity. RERANKER_MODEL defaults to BAAI/bge-reranker-base for
CPU speed; BAAI/bge-reranker-v2-m3 is more accurate (and multilingual) where a GPU or
more latency is available.
"""

import logging
import threading
from typing import Protocol

from app.core.config import get_settings

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)]


_lock = threading.Lock()
_reranker: Reranker | None = None
_disabled = False


def get_reranker() -> Reranker | None:
    """The shared reranker, or None when disabled or not loadable (retrieval then uses RRF order)."""
    global _reranker, _disabled
    s = get_settings()
    if not s.reranker_enabled or _disabled:
        return None
    with _lock:
        if _reranker is None:
            try:
                _reranker = CrossEncoderReranker(s.reranker_model)
            except Exception as e:
                log.error("Reranker %s could not be loaded (%s); falling back to fused ranking", s.reranker_model, e)
                _disabled = True
                return None
        return _reranker


def set_reranker(reranker: Reranker | None) -> None:
    global _reranker, _disabled
    with _lock:
        _reranker, _disabled = reranker, False
