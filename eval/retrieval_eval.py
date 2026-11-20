"""Retrieval evaluation for StayChat.

Compares dense-only, BM25-only, RRF hybrid, and the pipeline's actual
generation context (RRF plus support chunks) against eval/labels.jsonl.
No LLM is called and no API key is needed.

Run from the repo root:
    python -m eval.retrieval_eval            # use the cached index if current
    python -m eval.retrieval_eval --rebuild  # rebuild chunks and index first

Relevance is judged per source document. Rankings are lists of chunks,
mapped to their source document id; a document counts once, at its
best-ranked chunk.
"""
import argparse
import json
import logging
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import config
from eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

EVAL_DIR = Path(__file__).parent
LABELS_PATH = EVAL_DIR / "labels.jsonl"
RESULTS_JSON = EVAL_DIR / "results.json"
RESULTS_MD = EVAL_DIR / "results.md"
BASELINE_JSON = EVAL_DIR / "results_before_chunk_fix.json"

TOP_K = 10
METHODS = {
    "dense": "Dense only (MiniLM + FAISS)",
    "bm25": "BM25 only",
    "rrf": "RRF hybrid (k=60)",
    "rrf_support": "RRF + support chunks (pipeline context)",
}
METRICS = ["recall@5", "recall@10", "hit@5", "hit@10", "mrr@10", "ndcg@10"]
TYPE_ORDER = ["exact_name", "paraphrase", "multi_condition", "comparison", "policy"]


def load_labels(path: Path = LABELS_PATH) -> list[dict]:
    """Read labelled queries and check every relevant id exists in the corpus."""
    with open(config.DOCUMENTS_FILE, encoding="utf-8") as f:
        doc_ids = {d["id"] for d in json.load(f)}
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))
    seen = set()
    for item in labels:
        if item["id"] in seen:
            raise ValueError(f"duplicate query id {item['id']}")
        seen.add(item["id"])
        unknown = set(item["relevant_docs"]) - doc_ids
        if unknown:
            raise ValueError(f"{item['id']}: unknown doc ids {sorted(unknown)}")
        if not item["relevant_docs"]:
            raise ValueError(f"{item['id']}: no relevant docs")
    return labels


def score_ranking(ranked_docs: list[str], relevant: set[str]) -> dict:
    """Compute every metric for one ranking."""
    return {
        "recall@5": recall_at_k(ranked_docs, relevant, 5),
        "recall@10": recall_at_k(ranked_docs, relevant, 10),
        "hit@5": hit_at_k(ranked_docs, relevant, 5),
        "hit@10": hit_at_k(ranked_docs, relevant, 10),
        "mrr@10": reciprocal_rank(ranked_docs, relevant, 10),
        "ndcg@10": ndcg_at_k(ranked_docs, relevant, 10),
    }


def run_methods(pipeline, query: str) -> dict[str, list[str]]:
    """Return each method's ranking as a list of source document ids."""
    retriever = pipeline.retriever
    chunk_lists = {
        "dense": retriever.retrieve_dense(query, k=TOP_K),
        "bm25": retriever.retrieve_bm25(query, k=TOP_K),
        "rrf": retriever.retrieve_hybrid(query, k=TOP_K),
        "rrf_support": pipeline.build_context(query)["generation_chunks"],
    }
    return {name: [c["source_doc_id"] for c in chunks] for name, chunks in chunk_lists.items()}


def mean_metrics(rows: list[dict]) -> dict:
    """Average each metric over a list of per-query score dicts."""
    return {m: round(statistics.fmean(r[m] for r in rows), 4) for m in METRICS}


def aggregate(per_query: list[dict]) -> dict:
    """Overall, per-type and held-out (not tuned on) averages for each method."""
    out = {}
    for method in METHODS:
        rows = [(q, q["scores"][method]) for q in per_query]
        out[method] = {
            "overall": mean_metrics([s for _, s in rows]),
            "not_tuned_on": mean_metrics([s for q, s in rows if not q["tuned_on"]]),
            "by_type": {
                t: mean_metrics([s for q, s in rows if q["type"] == t])
                for t in TYPE_ORDER
                if any(q["type"] == t for q, _ in rows)
            },
        }
    return out


def git_commit() -> str:
    """Current commit hash, or 'unknown' outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=EVAL_DIR.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _row(label: str, scores: dict) -> str:
    return f"| {label} | " + " | ".join(f"{scores[m]:.3f}" for m in METRICS) + " |"


def _table(title_col: str, rows: list[tuple[str, dict]]) -> str:
    header = f"| {title_col} | Recall@5 | Recall@10 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    return "\n".join([header, sep] + [_row(label, s) for label, s in rows])


def render_markdown(results: dict) -> str:
    """Render results.json content as the results.md report."""
    meta = results["meta"]
    agg = results["aggregate"]
    n_tuned = meta["n_tuned_on"]
    lines = [
        "# Retrieval evaluation results",
        "",
        f"Generated by `python -m eval.retrieval_eval` on {meta['generated_at']} "
        f"(commit `{meta['git_commit']}`).",
        "",
        f"- Queries: {meta['n_queries']} hand-labelled (`eval/labels.jsonl`), "
        f"{meta['n_relevant_total']} relevant documents in total",
        f"- Corpus: {meta['n_documents']} documents, {meta['n_chunks']} chunks "
        f"(semantic chunking), embedding model `{meta['embedding_model']}`",
        f"- Relevance is judged per source document; labels were written by reading the corpus, "
        "not by looking at retriever output",
        "- Rankings are chunk lists mapped to source documents; a document is credited once, "
        "at its best-ranked chunk",
        "- Recall@k = share of a query's relevant documents found in the top k chunks; "
        "Hit@k = at least one relevant document in the top k; "
        "MRR@10 and NDCG@10 use binary relevance",
        "",
        "## Overall",
        "",
        _table("Method", [(METHODS[m], agg[m]["overall"]) for m in METHODS]),
        "",
        f"## Excluding the {n_tuned} queries the support-chunk rules were tuned on",
        "",
        "The keyword rules in `src/pipeline.py` (`topic_sets`, category bonuses) were hand-tuned "
        "earlier on a handful of demo queries, and those same queries are in the label set "
        f"(marked `tuned_on` in `labels.jsonl`). This table drops them ({meta['n_queries'] - n_tuned} "
        "queries left) so the support-chunk row is not scored on its own training examples.",
        "",
        _table("Method", [(METHODS[m], agg[m]["not_tuned_on"]) for m in METHODS]),
        "",
        "## By query type",
        "",
    ]
    for t in TYPE_ORDER:
        if t not in agg["rrf"]["by_type"]:
            continue
        n = sum(1 for q in results["per_query"] if q["type"] == t)
        lines += [f"### {t} ({n} queries)", "", _table("Method", [(METHODS[m], agg[m]["by_type"][t]) for m in METHODS]), ""]
    if BASELINE_JSON.exists():
        base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        lines += [
            "## Before the chunking fix",
            "",
            "The first run of this evaluation exposed a bug: the semantic chunker discarded every segment "
            "shorter than `MIN_CHUNK_TOKENS`, which silently removed about 9% of the corpus text "
            "(for example the sentence saying Coral Bay Suites allows free cancellation up to seven days "
            "before arrival). Short segments are now merged into a neighbouring chunk. "
            f"The same labels scored as follows on the old code ({base['meta']['n_chunks']} chunks, "
            f"raw numbers in `{BASELINE_JSON.name}`). The fix came from reading a failed query, not from "
            "tuning against the metric.",
            "",
            _table("Method", [(METHODS[m], base["aggregate"][m]["overall"]) for m in METHODS]),
            "",
        ]
    lines += ["## Queries with no relevant document in the top 5", ""]
    for m in METHODS:
        misses = [q for q in results["per_query"] if q["scores"][m]["hit@5"] == 0.0]
        names = ", ".join(f"`{q['id']}`" for q in misses) or "none"
        lines.append(f"- {METHODS[m]}: {len(misses)} of {meta['n_queries']} ({names})")
    lines += [
        "",
        "## Notes and caveats",
        "",
        "- Small label set (see counts above); differences of a few points are within noise.",
        "- List-style queries (for example `m02`, with 10 relevant documents) cannot reach "
        "Recall@5 = 1.0; Hit@5 and Recall@10 are the fairer numbers for them.",
        "- The `RRF + support` row scores the context the pipeline actually hands to the LLM: "
        f"{meta['generation_context_k']} RRF chunks plus support chunks, support first, capped at "
        f"{meta['max_generation_chunks']}. Support chunks come first, so rank-based metrics "
        "(Recall@5, Hit@5, MRR, NDCG) are penalised when a support chunk is irrelevant, and the cap can "
        "push RRF chunks out of the context. Recall@10 and Hit@10 do not depend on order and are the "
        "right columns for what the LLM sees.",
        "- BM25 and dense scores are not comparable; RRF ignores them and uses ranks only.",
        "- Answer quality is not measured here: that needs a live LLM and is not part of this script.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StayChat retrieval against labelled queries")
    parser.add_argument("--rebuild", action="store_true", help="rebuild chunks and FAISS index first")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--out-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--out-md", type=Path, default=RESULTS_MD)
    args = parser.parse_args()

    labels = load_labels(args.labels)

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(force_rebuild=args.rebuild, load_generator=False)
    logging.getLogger().setLevel(logging.WARNING)

    per_query = []
    for item in labels:
        relevant = set(item["relevant_docs"])
        rankings = run_methods(pipeline, item["query"])
        per_query.append(
            {
                "id": item["id"],
                "type": item["type"],
                "query": item["query"],
                "tuned_on": item["tuned_on"],
                "relevant_docs": item["relevant_docs"],
                "rankings": rankings,
                "scores": {m: score_ranking(r, relevant) for m, r in rankings.items()},
            }
        )

    with open(config.DOCUMENTS_FILE, encoding="utf-8") as f:
        n_documents = len(json.load(f))
    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "git_commit": git_commit(),
            "n_queries": len(labels),
            "n_tuned_on": sum(1 for q in labels if q["tuned_on"]),
            "n_relevant_total": sum(len(q["relevant_docs"]) for q in labels),
            "n_documents": n_documents,
            "n_chunks": len(pipeline.chunks),
            "embedding_model": config.EMBEDDING_MODEL,
            "rrf_k": config.RRF_K,
            "hybrid_candidate_k": config.HYBRID_CANDIDATE_K,
            "generation_context_k": config.GENERATION_CONTEXT_K,
            "max_generation_chunks": config.MAX_GENERATION_CHUNKS,
            "top_k": TOP_K,
        },
        "aggregate": aggregate(per_query),
        "per_query": per_query,
    }
    args.out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(results), encoding="utf-8")

    print(_table("Method", [(METHODS[m], results["aggregate"][m]["overall"]) for m in METHODS]))
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
