"""
RAG evaluation module for StayChat — retrieval metrics and qualitative analysis.

Metrics implemented:
    Precision@K  — fraction of retrieved top-K that are relevant.
    MRR          — Mean Reciprocal Rank; rewards systems that rank the
                   first relevant result highly.
    Qualitative  — per-query relevance and faithfulness verdicts derived
                   from heuristic signals in the retrieved chunks and answer.

Ground-truth relevance rules (derived from hotel_documents.json):
    Q1 (WiFi + breakfast):
        category == "amenities" AND hotel_name in
        {The Azure Grand, Sunrise Boutique Resort, Coral Bay Suites,
         Serenity Palms Resort}
    Q2 (Coral Bay cancellation):
        hotel_name == "Coral Bay Suites" AND category == "policies"
    Q3 (beach + excellent reviews):
        category in {"reviews", "location"} AND hotel_name in
        {Sunrise Boutique Resort, Serenity Palms Resort, Coral Bay Suites}

Failure-case documentation:
    "Tell me about the hotel" — deliberately vague; retrieves high-scoring
    chunks from multiple hotels across multiple categories because there is
    no discriminating signal. Precision@K collapses and the answer is
    either hallucinated or empty.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth definitions
# ---------------------------------------------------------------------------

# Hotels considered relevant for each query
_Q1_HOTELS = {"The Azure Grand", "Sunrise Boutique Resort", "Coral Bay Suites", "Serenity Palms Resort"}
_Q3_HOTELS = {"Sunrise Boutique Resort", "Serenity Palms Resort", "Coral Bay Suites"}

# Source doc IDs that satisfy each query's relevance criteria.
# Derived by walking hotel_documents.json and applying the rules above.
_GROUND_TRUTH_DOC_IDS: dict[str, set] = {
    # Q1: amenities docs for the four hotels that explicitly list free WiFi
    # and complimentary breakfast (hotel_009 Azure Grand amenities,
    # hotel_010 Sunrise amenities, hotel_011 Coral Bay amenities,
    # hotel_013 Serenity Palms amenities, hotel_014 Azure Grand extended
    # amenities, hotel_015 Coral Bay extended amenities).
    "Which hotels have free WiFi and complimentary breakfast?": {
        "hotel_009",  # The Azure Grand — amenities
        "hotel_010",  # Sunrise Boutique Resort — amenities
        "hotel_011",  # Coral Bay Suites — amenities
        "hotel_013",  # Serenity Palms Resort — amenities
        "hotel_014",  # The Azure Grand — amenities (extended)
        "hotel_015",  # Coral Bay Suites — amenities (extended)
    },
    # Q2: only the Coral Bay Suites policies document
    "What is the cancellation policy of Coral Bay Suites?": {
        "hotel_026",  # Coral Bay Suites — policies
    },
    # Q3: reviews and location docs for the three beach-focused hotels
    "Suggest a hotel with excellent reviews near the beach.": {
        "hotel_017",  # Sunrise Boutique Resort — reviews
        "hotel_018",  # Sunrise Boutique Resort — reviews
        "hotel_020",  # Coral Bay Suites — reviews
        "hotel_021",  # Coral Bay Suites — reviews
        "hotel_023",  # Serenity Palms Resort — reviews
        "hotel_024",  # Serenity Palms Resort — reviews
        "hotel_025",  # Serenity Palms Resort — reviews
        "hotel_032",  # Sunrise Boutique Resort — location
        "hotel_033",  # Serenity Palms Resort — location
        "hotel_035",  # Coral Bay Suites — location
        "hotel_040",  # Sunrise Boutique Resort — reviews
    },
}


def _relevant_doc_ids_for_query(query: str) -> set:
    """Return the set of relevant source_doc_ids for a known query.

    For unknown queries the empty set is returned, which will cause all
    metrics to score zero — intentional, since we have no ground truth.
    """
    return _GROUND_TRUTH_DOC_IDS.get(query, set())


def _is_chunk_relevant(chunk: dict, relevant_doc_ids: set) -> bool:
    """Return True when the chunk's source document is in the relevant set."""
    return chunk.get("source_doc_id", "") in relevant_doc_ids


class RAGEvaluator:
    """Evaluates retrieval and generation quality for the StayChat RAG pipeline."""

    def __init__(self, retriever=None, generator=None):
        """
        Initialize the evaluator.

        Args:
            retriever: HotelRetriever instance (used by run_full_evaluation).
            generator: HotelGenerator instance (used by run_full_evaluation).
        """
        self.retriever = retriever
        self.generator = generator

    def run_full_evaluation(self) -> None:
        """
        Run the complete evaluation suite using the injected retriever and generator.

        Builds queries_data for all DEMO_QUERIES plus the failure case,
        calls evaluate_all, and prints the formatted report.

        Returns:
            None. Prints formatted report to stdout.

        Example:
            >>> evaluator.run_full_evaluation()
        """
        queries_data = []
        all_queries = config.DEMO_QUERIES + ["Tell me about the hotel"]

        for query in all_queries:
            retrieved = self.retriever.retrieve(query, k=config.DEFAULT_K)
            filtered = self.retriever.filter_by_threshold(retrieved)
            answer_result = self.generator.generate(query, filtered)
            queries_data.append({
                "query": query,
                "retrieved_chunks": retrieved,
                "answer": answer_result["answer"],
            })

        results = self.evaluate_all(queries_data)
        self.print_evaluation_report(results)

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------

    def precision_at_k(self, retrieved_ids: list, relevant_ids: set, k: int) -> float:
        """
        Compute Precision@K: fraction of the top-K retrieved items that are relevant.

        Formula: |relevant ∩ retrieved[:k]| / k

        Args:
            retrieved_ids: Ordered list of retrieved source_doc_id strings,
                           most relevant first.
            relevant_ids:  Set of ground-truth relevant source_doc_id strings.
            k:             Cut-off depth.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 when k <= 0.

        Example:
            >>> ev = RAGEvaluator()
            >>> ev.precision_at_k(["hotel_009", "hotel_010", "hotel_001"], {"hotel_009", "hotel_010"}, k=3)
            0.6666666666666666
        """
        if k <= 0:
            logger.warning("precision_at_k called with k=%d; returning 0.0", k)
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        score = hits / k
        logger.debug("P@%d = %d/%d = %.4f", k, hits, k, score)
        return score

    def mrr(self, retrieved_ids: list, relevant_ids: set) -> float:
        """
        Compute Mean Reciprocal Rank (MRR) for a single query.

        Formula: 1 / rank_of_first_relevant  (0.0 if none found)

        Args:
            retrieved_ids: Ordered list of retrieved source_doc_id strings,
                           most relevant first (1-indexed internally).
            relevant_ids:  Set of ground-truth relevant source_doc_id strings.

        Returns:
            Float in (0.0, 1.0]. Returns 0.0 when no relevant item is found.

        Example:
            >>> ev = RAGEvaluator()
            >>> ev.mrr(["hotel_001", "hotel_026", "hotel_003"], {"hotel_026"})
            0.5
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                score = 1.0 / rank
                logger.debug("MRR first relevant at rank %d → %.4f", rank, score)
                return score
        logger.debug("MRR: no relevant document found in retrieved list → 0.0")
        return 0.0

    # ------------------------------------------------------------------
    # Qualitative analysis
    # ------------------------------------------------------------------

    def qualitative_analysis(self, query: str, retrieved_chunks: list, answer: str) -> dict:
        """
        Produce heuristic relevance and faithfulness verdicts for a single query.

        Relevance verdict: PASS when at least one retrieved chunk comes from a
        relevant source document; FAIL otherwise.

        Faithfulness verdict: PASS when the answer contains at least one bracketed
        citation of the form [hotel_XXX_chunk_N] and does NOT contain the
        fallback "I don't have enough information" phrase; PARTIAL when citations
        are present but the fallback phrase also appears; FAIL when no citations
        are found in the answer.

        Args:
            query:             The user question that was answered.
            retrieved_chunks:  List of chunk dicts returned by the retriever.
            answer:            The LLM-generated answer string.

        Returns:
            Dict with keys:
                relevance_verdict (str):    "PASS" | "FAIL"
                faithfulness_verdict (str): "PASS" | "PARTIAL" | "FAIL"
                relevant_chunks_found (int): count of retrieved chunks whose
                    source_doc_id is in the ground-truth relevant set.
                total_chunks_retrieved (int): len(retrieved_chunks)
                citations_in_answer (list[str]): chunk IDs cited in the answer.
                notes (str): human-readable explanation of the verdicts.

        Example:
            >>> ev = RAGEvaluator()
            >>> result = ev.qualitative_analysis(query, chunks, answer)
            >>> result["relevance_verdict"] in ("PASS", "FAIL")
            True
        """
        import re

        relevant_ids = _relevant_doc_ids_for_query(query)
        relevant_count = sum(
            1 for c in retrieved_chunks if _is_chunk_relevant(c, relevant_ids)
        )

        relevance_verdict = "PASS" if relevant_count > 0 else "FAIL"

        citation_pattern = re.compile(r'\[([^\]]+_chunk_\d+)\]')
        citations = citation_pattern.findall(answer)
        fallback_phrase = "i don't have enough information"
        answer_lower = answer.lower()

        if citations and fallback_phrase not in answer_lower:
            faithfulness_verdict = "PASS"
            notes = (
                f"Retrieval found {relevant_count}/{len(retrieved_chunks)} relevant chunks. "
                f"Answer cites {len(citations)} source(s) without fallback language."
            )
        elif citations and fallback_phrase in answer_lower:
            faithfulness_verdict = "PARTIAL"
            notes = (
                f"Retrieval found {relevant_count}/{len(retrieved_chunks)} relevant chunks. "
                f"Answer contains citations but also the fallback phrase, suggesting "
                f"partial context coverage."
            )
        else:
            faithfulness_verdict = "FAIL"
            notes = (
                f"Retrieval found {relevant_count}/{len(retrieved_chunks)} relevant chunks. "
                f"Answer contains no inline citations — cannot verify grounding."
            )

        logger.debug(
            "Qualitative: query='%s' relevance=%s faithfulness=%s",
            query[:60],
            relevance_verdict,
            faithfulness_verdict,
        )

        return {
            "relevance_verdict": relevance_verdict,
            "faithfulness_verdict": faithfulness_verdict,
            "relevant_chunks_found": relevant_count,
            "total_chunks_retrieved": len(retrieved_chunks),
            "citations_in_answer": citations,
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, queries_data: list) -> dict:
        """
        Run all metrics across every entry in queries_data.

        Each entry in queries_data must be a dict with:
            query (str):              the question asked.
            retrieved_chunks (list):  chunk dicts returned by the retriever.
            answer (str):             the LLM-generated answer.

        Returns:
            Dict with keys:
                per_query (list[dict]): one result dict per query, containing:
                    query, precision_at_k, mrr, qualitative, and a
                    failure_analysis field (non-empty only for known failure cases).
                aggregate (dict): mean_precision_at_k, mean_mrr computed over
                    all queries that have ground-truth data.

        Example:
            >>> results = ev.evaluate_all(queries_data)
            >>> "aggregate" in results and "per_query" in results
            True
        """
        per_query_results = []
        k = config.DEFAULT_K

        precision_scores = []
        mrr_scores = []

        for entry in queries_data:
            query = entry["query"]
            retrieved_chunks = entry.get("retrieved_chunks", [])
            answer = entry.get("answer", "")

            relevant_ids = _relevant_doc_ids_for_query(query)
            retrieved_doc_ids = [c.get("source_doc_id", "") for c in retrieved_chunks]

            p_at_k = self.precision_at_k(retrieved_doc_ids, relevant_ids, k=k)
            mrr_score = self.mrr(retrieved_doc_ids, relevant_ids)
            qualitative = self.qualitative_analysis(query, retrieved_chunks, answer)

            failure_analysis = ""
            if query == "Tell me about the hotel":
                failure_analysis = (
                    "DOCUMENTED FAILURE CASE: The query 'Tell me about the hotel' is "
                    "intentionally vague — it provides no hotel name, no category signal "
                    "(amenities / policies / reviews / location), and no specificity about "
                    "which property the user means. The embedding of this query sits near "
                    "the centroid of the entire corpus, so the FAISS retriever surfaces "
                    "high-scoring chunks from multiple unrelated hotels and categories. "
                    "Because no ground-truth relevant set exists for this query, "
                    "Precision@K and MRR both score 0.0. The LLM, faced with a broad "
                    "context from many hotels, either produces a generic summary that "
                    "cannot be verified against any single ground truth, or falls back to "
                    "the 'not enough information' response. In a production system, a "
                    "query clarification step (asking 'Which hotel are you asking about?') "
                    "would be inserted before retrieval."
                )
                logger.warning(
                    "Failure case query detected: '%s' — no ground truth; "
                    "P@K=%.2f MRR=%.2f",
                    query,
                    p_at_k,
                    mrr_score,
                )
            else:
                if relevant_ids:
                    precision_scores.append(p_at_k)
                    mrr_scores.append(mrr_score)

            logger.info(
                "Query='%s' P@%d=%.4f MRR=%.4f relevance=%s faithfulness=%s",
                query[:60],
                k,
                p_at_k,
                mrr_score,
                qualitative["relevance_verdict"],
                qualitative["faithfulness_verdict"],
            )

            per_query_results.append({
                "query": query,
                "precision_at_k": p_at_k,
                "k": k,
                "mrr": mrr_score,
                "qualitative": qualitative,
                "failure_analysis": failure_analysis,
            })

        aggregate = {
            "mean_precision_at_k": (
                sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
            ),
            "mean_mrr": (
                sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
            ),
            "queries_with_ground_truth": len(precision_scores),
            "total_queries": len(queries_data),
            "k": k,
        }

        logger.info(
            "Aggregate — mean P@%d=%.4f  mean MRR=%.4f  (%d/%d queries with GT)",
            k,
            aggregate["mean_precision_at_k"],
            aggregate["mean_mrr"],
            aggregate["queries_with_ground_truth"],
            aggregate["total_queries"],
        )

        return {"per_query": per_query_results, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # Report printing
    # ------------------------------------------------------------------

    def print_evaluation_report(self, results: dict) -> None:
        """
        Print a formatted evaluation report to stdout.

        Args:
            results: Dict returned by evaluate_all().

        Returns:
            None.

        Example:
            >>> ev.print_evaluation_report(results)
        """
        sep_heavy = "=" * 72
        sep_light = "-" * 72

        print(sep_heavy)
        print("  StayChat RAG — Evaluation Report")
        print(sep_heavy)

        for idx, pq in enumerate(results["per_query"], start=1):
            query = pq["query"]
            k = pq["k"]
            qual = pq["qualitative"]

            print(f"\nQuery {idx}: {query}")
            print(sep_light)
            print(f"  Precision@{k}             : {pq['precision_at_k']:.4f}")
            print(f"  MRR                      : {pq['mrr']:.4f}")
            print(
                f"  Relevant chunks retrieved : "
                f"{qual['relevant_chunks_found']} / {qual['total_chunks_retrieved']}"
            )
            print(f"  Relevance verdict        : {qual['relevance_verdict']}")
            print(f"  Faithfulness verdict     : {qual['faithfulness_verdict']}")

            if qual["citations_in_answer"]:
                print(f"  Citations in answer      : {', '.join(qual['citations_in_answer'])}")
            else:
                print("  Citations in answer      : (none)")

            print(f"  Notes                    : {qual['notes']}")

            if pq["failure_analysis"]:
                print()
                print("  *** FAILURE CASE ANALYSIS ***")
                # Word-wrap the failure analysis at 68 chars for readability
                words = pq["failure_analysis"].split()
                line = "  "
                for word in words:
                    if len(line) + len(word) + 1 > 70:
                        print(line)
                        line = "  " + word
                    else:
                        line = line + (" " if line.strip() else "") + word
                if line.strip():
                    print(line)

        agg = results["aggregate"]
        print()
        print(sep_heavy)
        print("  Aggregate Metrics")
        print(sep_heavy)
        print(f"  Queries evaluated (with ground truth) : {agg['queries_with_ground_truth']}")
        print(f"  Total queries                         : {agg['total_queries']}")
        print(f"  Mean Precision@{agg['k']}                      : {agg['mean_precision_at_k']:.4f}")
        print(f"  Mean MRR                              : {agg['mean_mrr']:.4f}")
        print(sep_heavy)
        print()
