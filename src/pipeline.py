"""End-to-end RAG pipeline orchestrator for StayChat hotel Q&A."""
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.preprocessor import HotelPreprocessor
from src.embedder import HotelEmbedder
from src.retriever import HotelRetriever
from src.generator import HotelGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: ingest → embed → retrieve → generate."""

    def __init__(self, force_rebuild: bool = False):
        """Initialize the pipeline. Loads cached index if available, else builds it."""
        self._setup_logging()
        logger.info("Initializing StayChat RAG Pipeline...")

        self.preprocessor = HotelPreprocessor()
        self.embedder = HotelEmbedder()
        self.generator = HotelGenerator()

        if not force_rebuild and self._index_is_current():
            logger.info("Loading cached FAISS index...")
            self.index, self.chunks = self.embedder.load_index()
        else:
            logger.info("Building FAISS index from scratch...")
            self._build_index()

        self.retriever = HotelRetriever(self.index, self.chunks, self.embedder)
        logger.info("Pipeline ready. %d chunks indexed.", len(self.chunks))

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format=config.LOG_FORMAT,
        )

    def _index_is_current(self) -> bool:
        """Return True when cached index artifacts are newer than the source corpus."""
        if not config.FAISS_INDEX_PATH.exists() or not config.CHUNKS_PICKLE_PATH.exists():
            return False
        docs_mtime = config.DOCUMENTS_FILE.stat().st_mtime
        index_mtime = config.FAISS_INDEX_PATH.stat().st_mtime
        chunks_mtime = config.CHUNKS_PICKLE_PATH.stat().st_mtime
        return min(index_mtime, chunks_mtime) >= docs_mtime

    def _build_index(self):
        """Load documents, preprocess, embed, and save FAISS index."""
        with open(config.DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
        logger.info("Loaded %d documents from %s", len(documents), config.DOCUMENTS_FILE)

        self.chunks = self.preprocessor.process_all(documents, self.embedder)
        embeddings = self.embedder.embed_chunks(self.chunks)
        self.index = self.embedder.build_faiss_index(embeddings)
        self.embedder.save_index(self.index, self.chunks)

    def _named_hotels(self, question: str) -> list[str]:
        """Return hotel names explicitly mentioned in the question."""
        question_lower = question.lower()
        hotel_names = sorted({chunk["hotel_name"] for chunk in self.chunks})
        return [name for name in hotel_names if name.lower() in question_lower]

    def _query_tokens(self, question: str) -> set[str]:
        normalized = question.lower().replace("wi-fi", "wifi")
        return set(re.findall(r"[a-z0-9]+", normalized))

    def _supporting_chunks_for_named_hotels(self, question: str, current_chunks: list) -> list:
        """Add high-overlap chunks for explicitly named hotels that retrieval under-covered."""
        named_hotels = self._named_hotels(question)
        if not named_hotels:
            return []

        current_ids = {chunk["chunk_id"] for chunk in current_chunks}
        query_tokens = self._query_tokens(question)
        added = []

        for hotel_name in named_hotels:
            candidates = []
            for chunk in self.chunks:
                if chunk["hotel_name"] != hotel_name or chunk["chunk_id"] in current_ids:
                    continue
                chunk_tokens = self._query_tokens(chunk["text"])
                overlap = len(query_tokens & chunk_tokens)
                category_bonus = 2 if chunk["category"] == "amenities" else 0
                exact_bonus = 1 if hotel_name.lower() in chunk["text"].lower() else 0
                score = overlap + category_bonus + exact_bonus
                if score > 0:
                    candidates.append((score, chunk))

            for _, chunk in sorted(candidates, key=lambda item: item[0], reverse=True)[:2]:
                enriched = dict(chunk)
                enriched["score"] = 0.0
                enriched["support_score"] = 0.0
                added.append(enriched)
                current_ids.add(enriched["chunk_id"])

        return added

    def _supporting_chunks_for_topic_terms(self, question: str, current_chunks: list) -> list:
        """Add canonical amenity/policy chunks for common multi-condition questions."""
        query_tokens = self._query_tokens(question)
        topic_sets = [
            {"wifi", "breakfast"},
            {"bathroom"},
            {"toiletries"},
            {"television"},
            {"televisions"},
            {"tv"},
            {"children"},
            {"pets"},
            {"airport"},
            {"layovers"},
        ]
        active_sets = [terms for terms in topic_sets if terms <= query_tokens]
        if not active_sets:
            return []

        current_ids = {chunk["chunk_id"] for chunk in current_chunks}
        candidates = []
        for chunk in self.chunks:
            if chunk["chunk_id"] in current_ids:
                continue
            chunk_tokens = self._query_tokens(chunk["text"])
            if not any(terms <= chunk_tokens for terms in active_sets):
                continue

            overlap = len(query_tokens & chunk_tokens)
            category_bonus = {
                "amenities": 6,
                "policies": 4,
                "description": 2,
                "location": 2,
                "reviews": 0,
            }.get(chunk["category"], 0)
            canonical_bonus = sum(
                1
                for word in ("complimentary", "included", "free", "provides", "policy", "amenities")
                if word in chunk_tokens
            )
            candidates.append((overlap + category_bonus + canonical_bonus, chunk))

        added = []
        per_hotel_counts: dict[str, int] = {}
        for _, chunk in sorted(candidates, key=lambda item: item[0], reverse=True):
            if len(added) >= 6:
                break
            # Keep list-style answers diverse instead of flooding context with one hotel.
            hotel_count = per_hotel_counts.get(chunk["hotel_name"], 0)
            if hotel_count >= 1 and len(per_hotel_counts) >= 3:
                continue
            enriched = dict(chunk)
            enriched["score"] = 0.0
            enriched["support_score"] = 0.0
            added.append(enriched)
            per_hotel_counts[enriched["hotel_name"]] = hotel_count + 1
            current_ids.add(enriched["chunk_id"])

        return added

    def _dedupe_and_cap_chunks(self, chunks: list) -> list:
        """Remove duplicate chunks and cap prompt context size."""
        deduped = []
        seen = set()
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid in seen:
                continue
            deduped.append(chunk)
            seen.add(cid)
        return deduped[:config.MAX_GENERATION_CHUNKS]

    def query(self, question: str) -> dict:
        """Run a full RAG query: retrieve → filter → generate.

        Returns a dict with keys: question, retrieved_chunks, filtered_chunks,
        answer, sources_cited, top_score.
        """
        start = time.perf_counter()
        retrieved = self.retriever.retrieve_hybrid(question, k=config.GENERATION_CONTEXT_K)
        filtered = self.retriever.filter_by_threshold(retrieved, min_score=0.0)
        support_start = time.perf_counter()
        support_chunks = self._supporting_chunks_for_topic_terms(question, filtered)
        support_chunks.extend(self._supporting_chunks_for_named_hotels(question, filtered + support_chunks))
        generation_chunks = self._dedupe_and_cap_chunks(support_chunks + filtered)
        generation = self.generator.generate(question, generation_chunks)
        elapsed = time.perf_counter() - start
        logger.info(
            "Query completed in %.2fs (retrieved=%d, support=%d, generation_chunks=%d, support_time=%.2fs)",
            elapsed,
            len(retrieved),
            len(support_chunks),
            len(generation_chunks),
            time.perf_counter() - support_start,
        )

        return {
            "question": question,
            "retrieved_chunks": retrieved,
            "filtered_chunks": generation_chunks,
            "answer": generation["answer"],
            "sources_cited": generation["sources_cited"],
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
        }
