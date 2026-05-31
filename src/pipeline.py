"""
End-to-end RAG pipeline orchestrator for StayChat hotel Q&A.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.preprocessor import HotelPreprocessor
from src.embedder import HotelEmbedder
from src.retriever import HotelRetriever
from src.generator import HotelGenerator
from src.evaluator import RAGEvaluator
from src.hallucination_control import HallucinationController

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: ingest → embed → retrieve → generate."""

    def __init__(self, force_rebuild: bool = False):
        """
        Initialize the full pipeline. Loads cached index if available, else builds it.

        Args:
            force_rebuild: If True, rebuild index even if cached version exists.
        """
        self._setup_logging()
        logger.info("Initializing StayChat RAG Pipeline...")

        self.preprocessor = HotelPreprocessor()
        self.embedder = HotelEmbedder()
        self.generator = HotelGenerator()

        if not force_rebuild and config.FAISS_INDEX_PATH.exists():
            logger.info("Loading cached FAISS index...")
            self.index, self.chunks = self.embedder.load_index()
        else:
            logger.info("Building FAISS index from scratch...")
            self._build_index()

        self.retriever = HotelRetriever(self.index, self.chunks, self.embedder)
        self.hallucination_controller = HallucinationController(self.retriever, self.generator)
        self.evaluator = RAGEvaluator(self.retriever, self.generator)

        logger.info("Pipeline ready. %d chunks indexed.", len(self.chunks))

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format=config.LOG_FORMAT,
        )

    def _build_index(self):
        """Load documents, preprocess, embed, and save FAISS index."""
        with open(config.DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
        logger.info("Loaded %d documents from %s", len(documents), config.DOCUMENTS_FILE)

        self.chunks = self.preprocessor.process_all(documents)
        embeddings = self.embedder.embed_chunks(self.chunks)
        self.index = self.embedder.build_faiss_index(embeddings)
        self.embedder.save_index(self.index, self.chunks)

    def query(self, question: str) -> dict:
        """
        Run a full RAG query: retrieve → filter → generate.

        Args:
            question: Natural language hotel question.

        Returns:
            Dict with keys:
                question (str)
                retrieved_chunks (list[dict]): top-k chunks with scores
                filtered_chunks (list[dict]): chunks above similarity threshold
                answer (str): LLM-generated answer
                sources_cited (list[str]): chunk IDs cited in the answer
                top_score (float): highest retrieval similarity score

        Example:
            >>> pipeline = RAGPipeline()
            >>> result = pipeline.query("Do you have free WiFi?")
            >>> "answer" in result
            True
        """
        retrieved = self.retriever.retrieve(question, k=config.DEFAULT_K)
        filtered = self.retriever.filter_by_threshold(retrieved)
        generation = self.generator.generate(question, filtered)

        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "filtered_chunks": filtered,
            "answer": generation["answer"],
            "sources_cited": generation["sources_cited"],
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
        }

    def run_demo_queries(self):
        """Run all three mandatory demo queries and print formatted results."""
        separator = "=" * 70
        print(f"\n{separator}")
        print("  STAYCHAT — MANDATORY DEMO QUERIES")
        print(separator)

        for i, question in enumerate(config.DEMO_QUERIES, 1):
            print(f"\n{'—' * 70}")
            print(f"  QUERY {i}: {question}")
            print(f"{'—' * 70}")

            result = self.query(question)

            print(f"\n  TOP RETRIEVED CHUNKS (k={config.DEFAULT_K}):")
            for j, chunk in enumerate(result["retrieved_chunks"], 1):
                preview = chunk["text"][:100].replace("\n", " ")
                print(f"  [{j}] {chunk['chunk_id']} | {chunk['hotel_name']} | "
                      f"{chunk['category']} | score={chunk['score']:.4f}")
                print(f"      \"{preview}...\"")

            print(f"\n  ANSWER:")
            print(f"  {result['answer']}")

            if result["sources_cited"]:
                print(f"\n  SOURCES CITED: {', '.join(result['sources_cited'])}")
            print(f"\n  CONFIDENCE (top score): {result['top_score']:.4f}")

        print(f"\n{separator}\n")

    def run_evaluation(self):
        """Run full evaluation suite and print report."""
        print("\n" + "=" * 70)
        print("  EVALUATION SUITE — Precision@k, MRR, Qualitative Analysis")
        print("=" * 70)
        self.evaluator.run_full_evaluation()

    def run_hallucination_demo(self):
        """Demonstrate hallucination control with before/after comparison."""
        print("\n" + "=" * 70)
        print("  HALLUCINATION CONTROL DEMONSTRATION")
        print("=" * 70)

        demo_query = config.DEMO_QUERIES[1]  # Cancellation policy — most fact-specific
        print(f"\n  Demo query: \"{demo_query}\"")

        print("\n  [1] STRICT PROMPT COMPARISON")
        print("  " + "-" * 40)
        comparison = self.hallucination_controller.strict_prompt_comparison(demo_query)
        print(f"\n  WITHOUT control:\n  {comparison['without_control']['answer'][:400]}")
        print(f"\n  WITH control:\n  {comparison['with_control']['answer'][:400]}")
        print(f"\n  Analysis:\n  {comparison['analysis']}")

        print("\n  [2] CONFIDENCE GATE — In-domain query (should pass)")
        gate_result = self.hallucination_controller.confidence_gate(demo_query)
        print(f"  Max score: {gate_result['max_score']:.4f} | Gate triggered: {gate_result['gate_triggered']}")
        print(f"  Message: {gate_result['message']}")

        print("\n  [3] CONFIDENCE GATE — Out-of-domain query (should trigger gate)")
        ood_query = "What is the current stock price of Marriott International?"
        gate_ood = self.hallucination_controller.confidence_gate(ood_query, threshold=0.5)
        print(f"  Query: \"{ood_query}\"")
        print(f"  Max score: {gate_ood['max_score']:.4f} | Gate triggered: {gate_ood['gate_triggered']}")
        print(f"  Message: {gate_ood['message']}")
