"""
Hotel chunk retriever — semantic search over FAISS index.

k=5 justified: balances recall vs. context window length.
Threshold filtering removes semantically distant chunks before generation.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.embedder import HotelEmbedder

logger = logging.getLogger(__name__)


class HotelRetriever:
    """Performs semantic search over the FAISS index of hotel chunks."""

    def __init__(self, index, chunks: list, embedder: HotelEmbedder):
        """
        Initialize the retriever with a loaded index and chunk list.

        Args:
            index: Loaded faiss.Index object.
            chunks: List of chunk dicts in the same order as index vectors.
            embedder: HotelEmbedder instance for query embedding.
        """
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    def retrieve(self, query: str, k: int = None) -> list:
        """
        Retrieve the top-k most semantically similar chunks for a query.

        Args:
            query: Natural language query string.
            k: Number of chunks to retrieve. Defaults to config.DEFAULT_K.

        Returns:
            List of chunk dicts augmented with a 'score' key,
            sorted descending by similarity score.

        Example:
            >>> results = retriever.retrieve("What hotels have free WiFi?", k=5)
            >>> results[0]["score"] >= results[-1]["score"]
            True
        """
        if k is None:
            k = config.DEFAULT_K

        query_vec = self.embedder.embed_query(query)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Retrieved %d chunks for query: '%s'", len(results), query[:60])
        return results

    def filter_by_threshold(self, results: list, min_score: float = None) -> list:
        """
        Drop chunks whose similarity score falls below the minimum threshold.

        Args:
            results: List of chunk dicts with 'score' key (from retrieve()).
            min_score: Minimum cosine similarity to keep. Defaults to
                       config.SIMILARITY_THRESHOLD.

        Returns:
            Filtered list of chunk dicts above the threshold.

        Example:
            >>> filtered = retriever.filter_by_threshold(results, min_score=0.3)
            >>> all(r["score"] >= 0.3 for r in filtered)
            True
        """
        if min_score is None:
            min_score = config.SIMILARITY_THRESHOLD

        filtered = [r for r in results if r["score"] >= min_score]
        dropped = len(results) - len(filtered)
        if dropped:
            logger.info(
                "Threshold filter (%.2f) dropped %d/%d chunks",
                min_score,
                dropped,
                len(results),
            )
        return filtered
