"""Qdrant vector store for chunk embeddings (cosine distance).

QDRANT_URL may be an http(s) URL (Docker / server), ":memory:" (tests) or a local folder
path (embedded mode, no server needed). Each point stores the chunk text and all of its
metadata as payload, so search results can be cited without another lookup.
"""

import logging
import uuid
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.rag.chunker import Chunk

log = logging.getLogger(__name__)
_NS = uuid.UUID("6f1c2a52-1d7e-4c2b-9d8e-3b5a7c9e1f20")


@lru_cache
def _client_for(url: str) -> QdrantClient:
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, timeout=60)
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=url)


class VectorStore:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None) -> None:
        s = get_settings()
        self.client = client or _client_for(s.qdrant_url)
        self.collection = collection or s.qdrant_collection

    def ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
            for fld in ("document_id", "chapter_key", "chunk_type"):
                try:
                    self.client.create_payload_index(self.collection, fld, models.PayloadSchemaType.KEYWORD)
                except Exception:  # embedded mode ignores payload indexes
                    pass

    def delete_document(self, document_id: str) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_doc_filter(document_id)))

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 128) -> None:
        self.ensure_collection(vectors.shape[1])
        for i in range(0, len(chunks), batch):
            pts = [models.PointStruct(id=str(uuid.uuid5(_NS, c.chunk_id)), vector=v.tolist(), payload=c.model_dump())
                   for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])]
            self.client.upsert(self.collection, points=pts, wait=True)

    def search(self, document_id: str, vector: np.ndarray, k: int, chapter_keys: list[str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.client.collection_exists(self.collection):
            return []
        flt = _doc_filter(document_id, chapter_keys)
        res = self.client.query_points(self.collection, query=vector.tolist(), query_filter=flt, limit=k, with_payload=True)
        return [(Chunk(**p.payload), float(p.score)) for p in res.points]


def _doc_filter(document_id: str, chapter_keys: list[str] | None = None) -> models.Filter:
    must: list = [models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    if chapter_keys:
        must.append(models.FieldCondition(key="chapter_key", match=models.MatchAny(any=chapter_keys)))
    return models.Filter(must=must)
