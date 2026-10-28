"""BM25 lexical retrieval over a document's chunks.

Catches what dense retrieval misses: exact figures, names, note numbers and terms such
as "OFS" or "GMP". The index is built from chunks.jsonl on first use and cached per
document.
"""

import re
import threading
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.rag.chunker import Chunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.,%/&-]*[a-z0-9%]|[a-z0-9]")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were which with".split())


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower().replace("₹", " inr "))
    # Index numbers both with and without separators so "1,234.5" matches "1234.5".
    out = []
    for t in toks:
        if t in _STOP:
            continue
        out.append(t)
        if any(ch.isdigit() for ch in t) and "," in t:
            out.append(t.replace(",", ""))
    return out


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.embedding_text()) for c in chunks]) if chunks else None

    def search(self, query: str, k: int, chapter_keys: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if chapter_keys and self.chunks[i].chapter_key not in chapter_keys:
                continue
            out.append((self.chunks[i], float(scores[i])))
            if len(out) >= k:
                break
        return out


_lock = threading.Lock()


@lru_cache(maxsize=16)
def _cached(document_id: str, mtime: float) -> BM25Index:
    from app.utils.storage import chunks_path

    with open(chunks_path(document_id), encoding="utf-8") as f:
        chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
    return BM25Index(chunks)


def get_bm25(document_id: str) -> BM25Index:
    from app.utils.storage import chunks_path

    p = chunks_path(document_id)
    if not p.exists():
        return BM25Index([])
    with _lock:
        return _cached(document_id, p.stat().st_mtime)
