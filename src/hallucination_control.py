"""
Hallucination control — strict prompt comparison and confidence gating.

Strict prompting removes the model's fallback to parametric memory.
Confidence gating prevents generation when retrieved context is
semantically distant from the query, catching out-of-domain questions
before they reach the LLM.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.generator import HotelGenerator
from src.retriever import HotelRetriever

logger = logging.getLogger(__name__)


class HallucinationController:
    """Implements two hallucination reduction strategies: strict prompting and confidence gating."""

    def __init__(self, retriever: HotelRetriever, generator: HotelGenerator):
        """
        Initialize with retriever and generator instances.

        Args:
            retriever: HotelRetriever for fetching context chunks.
            generator: HotelGenerator for LLM calls (both strict and unconstrained).
        """
        self.retriever = retriever
        self.generator = generator

    def strict_prompt_comparison(self, query: str) -> dict:
        """
        Run the same query twice — with and without hallucination controls — for side-by-side comparison.

        Args:
            query: Hotel question to compare.

        Returns:
            Dict with keys:
                query (str)
                chunks_used (list[dict]): retrieved chunks
                without_control (dict): answer from open-ended prompt
                with_control (dict): answer from strict context-only prompt
                analysis (str): human-readable explanation of differences

        Example:
            >>> ctrl = HallucinationController(retriever, generator)
            >>> result = ctrl.strict_prompt_comparison("What are the spa hours?")
            >>> "with_control" in result and "without_control" in result
            True
        """
        chunks = self.retriever.retrieve(query, k=config.DEFAULT_K)

        without_control = self.generator.generate_unconstrained(query, chunks)
        with_control = self.generator.generate(query, chunks)

        analysis = (
            "WITHOUT control: Model may draw on general hotel knowledge beyond the provided context, "
            "potentially inventing amenities, prices, or policies not in the dataset.\n"
            "WITH control: Model is restricted to only the retrieved context chunks. "
            "If the answer isn't in the context, it explicitly states it cannot answer."
        )

        logger.info("Strict prompt comparison complete for: '%s'", query[:60])
        return {
            "query": query,
            "chunks_used": chunks,
            "without_control": without_control,
            "with_control": with_control,
            "analysis": analysis,
        }

    def confidence_gate(self, query: str, threshold: float = None) -> dict:
        """
        Gate generation based on maximum retrieval similarity score.

        If the top-retrieved chunk's similarity score falls below threshold,
        refuse to generate rather than hallucinate from low-confidence context.

        Args:
            query: Hotel question to evaluate.
            threshold: Minimum acceptable similarity score. Defaults to
                       config.SIMILARITY_THRESHOLD.

        Returns:
            Dict with keys:
                query (str)
                max_score (float): highest similarity score from retrieval
                threshold (float): the gate threshold used
                gate_triggered (bool): True if generation was blocked
                result (dict | None): generation result if gate passed, else None
                message (str): explanation of what happened

        Example:
            >>> result = ctrl.confidence_gate("Tell me about politics")
            >>> result["gate_triggered"] == True  # out-of-domain query
            True
        """
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD

        chunks = self.retriever.retrieve(query, k=config.DEFAULT_K)

        if not chunks:
            return {
                "query": query,
                "max_score": 0.0,
                "threshold": threshold,
                "gate_triggered": True,
                "result": None,
                "message": "Insufficient context confidence. Cannot answer reliably.",
            }

        max_score = max(c["score"] for c in chunks)

        if max_score < threshold:
            logger.warning(
                "Confidence gate triggered for '%s' — max_score=%.3f < threshold=%.3f",
                query[:60],
                max_score,
                threshold,
            )
            return {
                "query": query,
                "max_score": max_score,
                "threshold": threshold,
                "gate_triggered": True,
                "result": None,
                "message": (
                    f"Insufficient context confidence. Cannot answer reliably. "
                    f"(Max similarity: {max_score:.3f}, required: {threshold:.3f})"
                ),
            }

        filtered_chunks = self.retriever.filter_by_threshold(chunks, threshold)
        result = self.generator.generate(query, filtered_chunks)

        return {
            "query": query,
            "max_score": max_score,
            "threshold": threshold,
            "gate_triggered": False,
            "result": result,
            "message": f"Gate passed. Max similarity: {max_score:.3f}",
        }
