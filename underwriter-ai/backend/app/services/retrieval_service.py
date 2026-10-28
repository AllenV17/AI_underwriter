"""Hybrid retrieval: BM25 + BGE-M3/Qdrant -> RRF -> cross-encoder rerank -> top chunks.

    query --> BM25 (top N) -----------\\
          --> embed -> Qdrant (top N) --> RRF -> top 10-20 -> reranker -> top 3-5

Every returned `Evidence` keeps the chunk's metadata, so answers cite real pages only.
"""

import logging
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.rag.bm25 import get_bm25
from app.rag.chunker import Chunk
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import get_reranker
from app.rag.vector_store import VectorStore
from app.services.embedding_service import get_embedder

log = logging.getLogger(__name__)


@dataclass
class Evidence:
    ref: str  # "S1", "S2"... as shown to the LLM
    chunk: Chunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return f"Source: DRHP, {self.chunk.section_label}, page {self.chunk.page_number}"

    def to_dict(self) -> dict:
        c = self.chunk
        return {"ref": self.ref, "page_number": c.page_number, "section": c.section_label, "chapter": c.chapter,
                "subsection": c.subsection, "chunk_type": c.chunk_type, "text": c.text,
                "dense_rank": self.dense_rank, "sparse_rank": self.sparse_rank,
                "rrf_score": round(self.rrf_score, 5),
                "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4)}


@dataclass
class RetrievalResult:
    query: str
    evidence: list[Evidence] = field(default_factory=list)
    candidates: int = 0
    reranked: bool = False


class RetrievalService:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.settings = get_settings()
        self.store = store or VectorStore()

    def retrieve(self, document_id: str, query: str, *, final_k: int | None = None,
                 chapter_keys: list[str] | None = None) -> RetrievalResult:
        s = self.settings
        final_k = final_k or s.retrieval_final_k

        sparse = get_bm25(document_id).search(query, s.retrieval_sparse_k, chapter_keys)
        try:
            qvec = get_embedder().embed_query(query)
            dense = self.store.search(document_id, qvec, s.retrieval_dense_k, chapter_keys)
        except Exception as e:  # retrieval still works lexically if the vector side is down
            log.error("Dense retrieval failed (%s); using BM25 only", e)
            dense = []

        by_id: dict[str, Chunk] = {}
        dense_rank, sparse_rank = {}, {}
        for i, (c, _) in enumerate(dense, 1):
            by_id[c.chunk_id] = c
            dense_rank[c.chunk_id] = i
        for i, (c, _) in enumerate(sparse, 1):
            by_id.setdefault(c.chunk_id, c)
            sparse_rank[c.chunk_id] = i

        fused = reciprocal_rank_fusion([[c.chunk_id for c, _ in dense], [c.chunk_id for c, _ in sparse]], k=s.rrf_k)
        fused = fused[: s.retrieval_fused_k]
        items = [Evidence(ref="", chunk=by_id[cid], dense_rank=dense_rank.get(cid), sparse_rank=sparse_rank.get(cid),
                          rrf_score=score) for cid, score in fused]

        reranker = get_reranker()
        if reranker and items:
            scores = reranker.score(query, [e.chunk.embedding_text() for e in items])
            for e, sc in zip(items, scores):
                e.rerank_score = sc
            items.sort(key=lambda e: e.rerank_score, reverse=True)

        items = items[:final_k]
        for i, e in enumerate(items, 1):
            e.ref = f"S{i}"
        return RetrievalResult(query=query, evidence=items, candidates=len(fused), reranked=bool(reranker and items))
