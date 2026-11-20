"""Tests for the ranking metrics and the label file used by eval/retrieval_eval.py."""
import math

import pytest

from eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from eval.retrieval_eval import aggregate, load_labels, score_ranking


def test_recall_counts_unique_relevant_docs_in_top_k():
    ranked = ["a", "b", "a", "c", "d"]
    assert recall_at_k(ranked, {"a", "d", "z"}, 3) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, {"a", "d", "z"}, 5) == pytest.approx(2 / 3)


def test_hit_at_k():
    assert hit_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert hit_at_k(["x", "y", "a"], {"a"}, 3) == 1.0


def test_reciprocal_rank_uses_first_relevant_and_respects_cutoff():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["x"] * 10 + ["a"], {"a"}, k=10) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_ndcg_perfect_and_worst_orderings():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}) == 0.0


def test_ndcg_known_value():
    # relevant at ranks 2 and 3 of 3 ranked, two relevant docs
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}) == pytest.approx(dcg / idcg)


def test_ndcg_does_not_double_count_repeated_docs():
    assert ndcg_at_k(["a", "a", "a"], {"a"}) == pytest.approx(1.0)


def test_empty_relevant_set_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 5)


def test_labels_file_is_valid():
    labels = load_labels()
    assert len(labels) >= 50
    assert {q["type"] for q in labels} == {"exact_name", "paraphrase", "multi_condition", "comparison", "policy"}


def test_aggregate_splits_tuned_queries():
    def entry(qid, tuned, hit):
        s = {"recall@5": hit, "recall@10": hit, "hit@5": hit, "hit@10": hit, "mrr@10": hit, "ndcg@10": hit}
        return {"id": qid, "type": "policy", "tuned_on": tuned,
                "scores": {m: s for m in ("dense", "bm25", "rrf", "rrf_support")}}

    agg = aggregate([entry("a", True, 1.0), entry("b", False, 0.0)])
    assert agg["rrf"]["overall"]["hit@5"] == 0.5
    assert agg["rrf"]["not_tuned_on"]["hit@5"] == 0.0
    assert agg["rrf"]["by_type"]["policy"]["hit@5"] == 0.5


def test_score_ranking_has_all_metrics():
    scores = score_ranking(["a", "b"], {"a"})
    assert set(scores) == {"recall@5", "recall@10", "hit@5", "hit@10", "mrr@10", "ndcg@10"}
