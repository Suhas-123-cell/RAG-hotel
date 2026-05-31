"""Hotel chunk retriever — hybrid BM25 + dense FAISS search with RRF merge."""
import logging
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.embedder import HotelEmbedder

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 while normalizing common hotel-query variants."""
    normalized = text.lower().replace("wi-fi", "wifi")
    return re.findall(r"[a-z0-9]+", normalized)


class HotelRetriever:
    """Hybrid BM25 + FAISS retriever with Reciprocal Rank Fusion."""

    def __init__(self, index, chunks: list, embedder: HotelEmbedder):
        """Initialize retriever and build the in-memory BM25 index."""
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

        logger.info("Building BM25 index over %d chunks...", len(chunks))
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=config.BM25_K1, b=config.BM25_B)
        logger.info("BM25 index ready")

    def retrieve_dense(self, query: str, k: int | None = None) -> list:
        """Retrieve top-k chunks by cosine similarity (FAISS)."""
        if k is None:
            k = config.HYBRID_CANDIDATE_K

        query_vec = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["dense_score"] = float(score)
            results.append(chunk)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.debug("Dense: %d results for '%s'", len(results), query[:50])
        return results

    def retrieve_bm25(self, query: str, k: int | None = None) -> list:
        """Retrieve top-k chunks by BM25 term-frequency scoring."""
        if k is None:
            k = config.HYBRID_CANDIDATE_K

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(scores[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        logger.debug("BM25: %d results for '%s'", len(results), query[:50])
        return results

    def retrieve_hybrid(self, query: str, k: int | None = None) -> list:
        """Retrieve top-k chunks using hybrid BM25 + FAISS with RRF merge."""
        if k is None:
            k = config.DEFAULT_K

        dense_results = self.retrieve_dense(query, k=config.HYBRID_CANDIDATE_K)
        sparse_results = self.retrieve_bm25(query, k=config.HYBRID_CANDIDATE_K)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            chunk_map[cid] = result

        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = result
            else:
                chunk_map[cid]["bm25_score"] = result.get("bm25_score", 0.0)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        final = []
        for cid, rrf_score in ranked:
            chunk = dict(chunk_map[cid])
            chunk["score"] = rrf_score
            chunk["rrf_score"] = rrf_score
            final.append(chunk)

        logger.info("Hybrid RRF: %d results for '%s'", len(final), query[:50])
        return final

    def retrieve(self, query: str, k: int | None = None) -> list:
        """Alias for retrieve_dense — kept for backwards compatibility with tests."""
        return self.retrieve_dense(query, k=k if k is not None else config.DEFAULT_K)

    def filter_by_threshold(self, results: list, min_score: float | None = None) -> list:
        """Drop chunks whose score falls below the similarity threshold."""
        if min_score is None:
            if results and results[0].get("rrf_score"):
                min_score = 1.0 / (config.RRF_K + 1)  # ~0.0164 — chunk must rank well in at least one system
            else:
                min_score = config.SIMILARITY_THRESHOLD

        filtered = [r for r in results if r["score"] >= min_score]
        dropped = len(results) - len(filtered)
        if dropped:
            logger.info("Threshold filter dropped %d/%d chunks (floor=%.4f)", dropped, len(results), min_score)
        return filtered
