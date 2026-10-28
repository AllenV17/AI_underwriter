"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum over rankings of 1 / (k + rank_d), with rank starting at 1. RRF uses ranks
only, so BM25 scores and cosine similarities need no normalisation to be combined.
"""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists; returns (id, fused score) sorted best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
