"""
Hotel chunk retriever — hybrid BM25 + dense FAISS search with RRF merge.

Retrieval strategy:
    1. Dense (FAISS): cosine similarity via all-MiniLM-L6-v2 embeddings.
       Catches semantic paraphrases — "complimentary internet" matches
       "free WiFi" even with zero lexical overlap.
    2. Sparse (BM25): term-frequency/IDF scoring over tokenised chunk text.
       Catches exact-match signals — hotel names, room numbers, policy
       keywords — that dense search may under-weight.
    3. Reciprocal Rank Fusion (RRF): merges both ranked lists without
       needing score normalisation. Score = Σ 1/(RRF_K + rank_i).

k=5 final results: balances recall vs. context window length.
Threshold filtering (dense score gate) removes low-confidence chunks.
"""
import logging
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.embedder import HotelEmbedder

logger = logging.getLogger(__name__)


class HotelRetriever:
    """Hybrid BM25 + FAISS retriever with Reciprocal Rank Fusion."""

    def __init__(self, index, chunks: list, embedder: HotelEmbedder):
        """
        Initialize retriever and build the in-memory BM25 index.

        Args:
            index: Loaded faiss.Index object.
            chunks: List of chunk dicts in the same order as index vectors.
            embedder: HotelEmbedder instance for query embedding.
        """
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

        logger.info("Building BM25 index over %d chunks...", len(chunks))
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=config.BM25_K1, b=config.BM25_B)
        logger.info("BM25 index ready")

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def retrieve_dense(self, query: str, k: int = None) -> list:
        """
        Retrieve top-k chunks by cosine similarity (FAISS).

        Args:
            query: Natural language query string.
            k: Number of results. Defaults to config.HYBRID_CANDIDATE_K.

        Returns:
            List of chunk dicts with 'score' (cosine similarity) added,
            sorted descending.

        Example:
            >>> results = retriever.retrieve_dense("free WiFi", k=10)
            >>> results[0]["score"] >= results[-1]["score"]
            True
        """
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

    # ------------------------------------------------------------------
    # Sparse retrieval (BM25)
    # ------------------------------------------------------------------

    def retrieve_bm25(self, query: str, k: int = None) -> list:
        """
        Retrieve top-k chunks by BM25 term-frequency scoring.

        Excels at exact keyword matches: hotel names, room types, policy
        terms, specific numbers (e.g. "48 hour", "room 204").

        Args:
            query: Natural language query string.
            k: Number of results. Defaults to config.HYBRID_CANDIDATE_K.

        Returns:
            List of chunk dicts with 'bm25_score' added, sorted descending.
            Chunks with score 0 are excluded.

        Example:
            >>> results = retriever.retrieve_bm25("Coral Bay Suites cancellation", k=10)
            >>> all("bm25_score" in r for r in results)
            True
        """
        if k is None:
            k = config.HYBRID_CANDIDATE_K

        tokenized_query = query.lower().split()
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

    # ------------------------------------------------------------------
    # Hybrid retrieval (BM25 + Dense + RRF)
    # ------------------------------------------------------------------

    def retrieve_hybrid(self, query: str, k: int = None) -> list:
        """
        Retrieve top-k chunks using hybrid BM25 + FAISS with RRF merge.

        Both systems contribute HYBRID_CANDIDATE_K candidates each. RRF
        combines the ranked lists without score normalisation:
            rrf_score(d) = Σ 1 / (RRF_K + rank_i(d))

        Args:
            query: Natural language query string.
            k: Final number of results after RRF merge. Defaults to
               config.DEFAULT_K.

        Returns:
            List of chunk dicts with 'score' (RRF), 'dense_score', and
            'bm25_score' populated where available, sorted by RRF score.

        Example:
            >>> results = retriever.retrieve_hybrid("free WiFi breakfast", k=5)
            >>> len(results) <= 5
            True
        """
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

    # keep dense-only method as `retrieve` for backwards compat with tests
    def retrieve(self, query: str, k: int = None) -> list:
        """
        Alias for retrieve_dense. Kept for backwards compatibility.

        Args:
            query: Natural language query string.
            k: Number of results.

        Returns:
            Dense retrieval results (see retrieve_dense).
        """
        return self.retrieve_dense(query, k=k if k is not None else config.DEFAULT_K)

    # ------------------------------------------------------------------
    # Threshold filter
    # ------------------------------------------------------------------

    def filter_by_threshold(self, results: list, min_score: float = None) -> list:
        """
        Drop chunks whose 'score' falls below the similarity threshold.

        For dense results 'score' is cosine similarity. For hybrid results
        it is the RRF score; the threshold is applied as a relative floor
        (any chunk below 1/(RRF_K*2) is dropped as a near-zero contributor).

        Args:
            results: List of chunk dicts with 'score' key.
            min_score: Floor score. Defaults to config.SIMILARITY_THRESHOLD
                       for dense; auto-scaled for RRF scores.

        Returns:
            Filtered list.

        Example:
            >>> filtered = retriever.filter_by_threshold(results)
            >>> len(filtered) <= len(results)
            True
        """
        if min_score is None:
            if results and results[0].get("rrf_score"):
                min_score = 1.0 / (config.RRF_K * 2)
            else:
                min_score = config.SIMILARITY_THRESHOLD

        filtered = [r for r in results if r["score"] >= min_score]
        dropped = len(results) - len(filtered)
        if dropped:
            logger.info("Threshold filter dropped %d/%d chunks (floor=%.4f)", dropped, len(results), min_score)
        return filtered
