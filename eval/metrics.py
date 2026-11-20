"""Ranking metrics over document ids.

A ranking is a list of document ids in rank order. It can contain repeats
because several chunks of one document may be retrieved; a document is only
credited the first time it appears.
"""
import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k positions."""
    if not relevant:
        raise ValueError("relevant set is empty")
    return len(relevant & set(ranked[:k])) / len(relevant)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document appears in the top k positions, else 0.0."""
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """1 / rank of the first relevant document within the top k, else 0.0."""
    for pos, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k. Repeated documents earn gain only at first appearance."""
    if not relevant:
        raise ValueError("relevant set is empty")
    seen: set[str] = set()
    dcg = 0.0
    for pos, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant and doc_id not in seen:
            dcg += 1.0 / math.log2(pos + 1)
        seen.add(doc_id)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal
