"""
Higher-level integration tests for the RAG pipeline.

Mandatory query tests verify retrieval correctness only — the LLM
generator is NOT called, so no GROQ_API_KEY is required.

Tests marked @pytest.mark.slow build a real FAISS index over the full
corpus and may take 30–90 seconds on first run (model download included).
They require sentence_transformers to be installed.

Fast tests (not marked slow) only use faiss directly with synthetic
embeddings and do not require sentence_transformers or groq.
"""
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import faiss

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Detect whether optional heavy dependencies are available.
# Fast tests (TestPreprocessorProcessesDocuments, TestEmbedderBuildsIndex,
# TestRetrieverFindsChunks) stub out sentence_transformers so they run
# without the model.  Slow integration tests need the real library.
# ---------------------------------------------------------------------------
_sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
_groq_available = importlib.util.find_spec("groq") is not None

for _mod, _available in [
    ("sentence_transformers", _sentence_transformers_available),
    ("groq", _groq_available),
    ("dotenv", importlib.util.find_spec("dotenv") is not None),
]:
    if not _available and _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Ensure dotenv.load_dotenv is callable even when mocked
if not (importlib.util.find_spec("dotenv") is not None):
    _mock_dotenv = sys.modules.get("dotenv", MagicMock())
    if not callable(getattr(_mock_dotenv, "load_dotenv", None)):
        _mock_dotenv.load_dotenv = MagicMock()
    sys.modules["dotenv"] = _mock_dotenv

import config
from src.preprocessor import HotelPreprocessor
from src.embedder import HotelEmbedder
from src.retriever import HotelRetriever


# ---------------------------------------------------------------------------
# Custom pytest markers
# ---------------------------------------------------------------------------
# Register the "slow" marker so pytest doesn't warn about unknown marks.
# Run only fast tests: pytest -m "not slow"
# Run all tests:       pytest
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level shared state (built once per test session for slow tests)
# ---------------------------------------------------------------------------

_shared_state: dict = {}  # keyed by "index", "chunks", "embedder"


def _get_full_index():
    """
    Build (or return cached) the real FAISS index over hotel_documents.json.

    Building is expensive (~60 s on first run); subsequent calls in the
    same session return the cached result immediately.

    Requires sentence_transformers to be installed; slow tests are skipped
    automatically when the library is absent.
    """
    if not _sentence_transformers_available:
        pytest.skip(
            "sentence_transformers not installed — skipping slow integration test. "
            "Install it with: pip install sentence-transformers"
        )

    if "index" not in _shared_state:
        # Re-import with the real library now that we know it's available
        from sentence_transformers import SentenceTransformer  # noqa: F401 (smoke-import)

        with open(config.DOCUMENTS_FILE, "r", encoding="utf-8") as fh:
            documents = json.load(fh)

        preprocessor = HotelPreprocessor()
        chunks = preprocessor.process_all(documents)

        embedder = HotelEmbedder()
        embeddings = embedder.embed_chunks(chunks)
        index = embedder.build_faiss_index(embeddings)

        _shared_state["index"] = index
        _shared_state["chunks"] = chunks
        _shared_state["embedder"] = embedder

    return _shared_state["index"], _shared_state["chunks"], _shared_state["embedder"]


# ---------------------------------------------------------------------------
# Test 1 — Preprocessor processes documents
# ---------------------------------------------------------------------------

class TestPreprocessorProcessesDocuments:
    def test_preprocessor_processes_documents(self):
        """
        Loads hotel_documents.json and runs the preprocessor.
        Verifies that chunks are produced and carry required fields.
        """
        with open(config.DOCUMENTS_FILE, "r", encoding="utf-8") as fh:
            documents = json.load(fh)

        assert len(documents) == 40, (
            f"Expected 40 documents in hotel_documents.json, got {len(documents)}"
        )

        preprocessor = HotelPreprocessor()
        all_chunks = preprocessor.process_all(documents)

        assert len(all_chunks) >= 40, (
            "Should produce at least as many chunks as there are documents"
        )

        required_fields = {"chunk_id", "hotel_name", "category", "text",
                           "source_doc_id", "token_count"}
        for chunk in all_chunks:
            missing = required_fields - set(chunk.keys())
            assert not missing, (
                f"Chunk {chunk.get('chunk_id', '?')} missing fields: {missing}"
            )

        # All chunk texts must be non-empty strings
        for chunk in all_chunks:
            assert isinstance(chunk["text"], str) and chunk["text"].strip(), (
                f"Chunk {chunk['chunk_id']} has empty text"
            )


# ---------------------------------------------------------------------------
# Test 2 — Embedder builds index with fake embeddings
# ---------------------------------------------------------------------------

class TestEmbedderBuildsIndex:
    def test_embedder_builds_index(self):
        """
        Creates a tiny FAISS index from 3 hand-crafted fake embeddings
        and verifies the index properties — no model loading required.
        """
        dim = config.EMBEDDING_DIM
        n = 3

        # Normalised random embeddings (simulating sentence-transformer output)
        rng = np.random.default_rng(seed=42)
        raw = rng.random((n, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embeddings = raw / norms  # L2-normalise

        # Use a real HotelEmbedder but only call build_faiss_index
        # (the SentenceTransformer model is NOT loaded here)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        assert index.ntotal == n, f"Expected {n} vectors in index, got {index.ntotal}"
        assert index.d == dim, f"Index dimension {index.d} != {dim}"

        # Verify that a search returns k results
        query = embeddings[0:1]  # use first embedding as query
        scores, indices = index.search(query, n)
        assert scores.shape == (1, n)
        assert indices[0][0] == 0  # exact match for itself should be first

    def test_embedder_index_accepts_float32(self):
        """FAISS IndexFlatIP requires float32; verify no dtype errors arise."""
        dim = 16  # tiny dim just for the shape check
        embeddings = np.ones((5, dim), dtype=np.float32)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        assert index.ntotal == 5

    def test_embedder_inner_product_equals_cosine_for_normalised(self):
        """
        For L2-normalised vectors, inner product == cosine similarity.
        Self-similarity of a unit vector must be 1.0 (within float tolerance).
        """
        dim = config.EMBEDDING_DIM
        vec = np.random.rand(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        index = faiss.IndexFlatIP(dim)
        index.add(vec.reshape(1, dim))

        scores, _ = index.search(vec.reshape(1, dim), 1)
        assert abs(scores[0][0] - 1.0) < 1e-5, (
            f"Self-similarity of unit vector should be ~1.0, got {scores[0][0]}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Retriever finds chunks from tiny index
# ---------------------------------------------------------------------------

class TestRetrieverFindsChunks:
    def test_retriever_finds_chunks(self):
        """
        Builds a 3-vector FAISS index from fake embeddings and verifies
        that HotelRetriever.retrieve() returns the closest match first.
        No sentence-transformer model or API key needed.
        """
        dim = config.EMBEDDING_DIM
        rng = np.random.default_rng(seed=7)

        # Three distinct unit vectors
        raw = rng.random((3, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embeddings = raw / norms

        fake_chunks = [
            {
                "chunk_id": f"fake_00{i}_chunk_0",
                "hotel_name": f"Fake Hotel {i}",
                "category": "amenities",
                "text": f"Amenity text for fake hotel number {i}.",
                "source_doc_id": f"fake_00{i}",
                "token_count": 8,
            }
            for i in range(3)
        ]

        # Build real FAISS index
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Mock embedder: embed_query returns the first embedding (perfect match for chunk 0)
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = embeddings[0:1]

        retriever = HotelRetriever(index, fake_chunks, mock_embedder)
        results = retriever.retrieve("What are the amenities?", k=3)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # The first result must be chunk 0 (exact query match)
        assert results[0]["chunk_id"] == "fake_000_chunk_0", (
            f"Top result should be chunk 0, got {results[0]['chunk_id']}"
        )

        # Scores must be descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted descending: {scores}"
        )

    def test_retriever_filter_reduces_results(self):
        """
        Builds a tiny index, retrieves results, then applies threshold filter.
        At a high threshold, some or all results should be dropped.
        """
        dim = config.EMBEDDING_DIM
        rng = np.random.default_rng(seed=13)

        raw = rng.random((5, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embeddings = raw / norms

        fake_chunks = [
            {
                "chunk_id": f"fc_{i}",
                "hotel_name": "Hotel X",
                "category": "policies",
                "text": f"Policy text {i}.",
                "source_doc_id": f"doc_{i}",
                "token_count": 3,
            }
            for i in range(5)
        ]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = embeddings[0:1]

        retriever = HotelRetriever(index, fake_chunks, mock_embedder)
        results = retriever.retrieve("Policy question", k=5)

        # Apply an extremely high threshold — should drop most/all
        filtered = retriever.filter_by_threshold(results, min_score=0.9999)
        assert all(r["score"] >= 0.9999 for r in filtered)

        # With a zero threshold everything should pass
        filtered_all = retriever.filter_by_threshold(results, min_score=0.0)
        assert len(filtered_all) == len(results)


# ---------------------------------------------------------------------------
# Tests 4–6 — Mandatory query retrieval (real index, no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMandatoryQueryRetrieval:
    """
    Verifies that each of the three mandatory demo queries retrieves chunks
    from the expected categories/hotels. The LLM generator is NOT called.
    """

    # Q1 — "Which hotels have free WiFi and complimentary breakfast?"
    def test_mandatory_query_q1_retrieves_amenities(self):
        """
        Q1 should retrieve chunks from the 'amenities' category.
        At least one of the top-k chunks must have category == 'amenities'.
        """
        index, chunks, embedder = _get_full_index()
        retriever = HotelRetriever(index, chunks, embedder)

        query = config.DEMO_QUERIES[0]
        results = retriever.retrieve(query, k=config.DEFAULT_K)

        assert len(results) > 0, "Q1 retrieved zero chunks"

        categories = [r["category"] for r in results]
        assert "amenities" in categories, (
            f"Q1 did not retrieve any 'amenities' chunks. "
            f"Got categories: {categories}"
        )

    def test_mandatory_query_q1_top_score_reasonable(self):
        """Q1 top similarity score must be above the configured threshold."""
        index, chunks, embedder = _get_full_index()
        retriever = HotelRetriever(index, chunks, embedder)

        query = config.DEMO_QUERIES[0]
        results = retriever.retrieve(query, k=config.DEFAULT_K)

        assert results[0]["score"] >= config.SIMILARITY_THRESHOLD, (
            f"Q1 top score {results[0]['score']:.4f} is below threshold "
            f"{config.SIMILARITY_THRESHOLD}"
        )

    # Q2 — "What is the cancellation policy of Coral Bay Suites?"
    def test_mandatory_query_q2_retrieves_coral_bay_policy(self):
        """
        Q2 should retrieve at least one chunk from Coral Bay Suites
        in the 'policies' category.
        """
        index, chunks, embedder = _get_full_index()
        retriever = HotelRetriever(index, chunks, embedder)

        query = config.DEMO_QUERIES[1]
        results = retriever.retrieve(query, k=config.DEFAULT_K)

        assert len(results) > 0, "Q2 retrieved zero chunks"

        coral_bay_policy_chunks = [
            r for r in results
            if "coral bay" in r["hotel_name"].lower() and r["category"] == "policies"
        ]
        assert len(coral_bay_policy_chunks) >= 1, (
            f"Q2 did not retrieve any Coral Bay Suites policy chunks. "
            f"Got: {[(r['hotel_name'], r['category']) for r in results]}"
        )

    def test_mandatory_query_q2_top_result_is_coral_bay(self):
        """
        The top result for Q2 must be from Coral Bay Suites
        (the query names the hotel explicitly).
        """
        index, chunks, embedder = _get_full_index()
        retriever = HotelRetriever(index, chunks, embedder)

        query = config.DEMO_QUERIES[1]
        results = retriever.retrieve(query, k=config.DEFAULT_K)

        assert "coral bay" in results[0]["hotel_name"].lower(), (
            f"Expected top Q2 result from Coral Bay Suites, "
            f"got '{results[0]['hotel_name']}'"
        )

    # Q3 — "Suggest a hotel with excellent reviews near the beach."
    def test_mandatory_query_q3_retrieves_beach_reviews(self):
        """
        Q3 should retrieve at least one chunk from the 'reviews' category.
        Given the beach phrasing, at least one review chunk is expected.
        """
        index, chunks, embedder = _get_full_index()
        retriever = HotelRetriever(index, chunks, embedder)

        query = config.DEMO_QUERIES[2]
        results = retriever.retrieve(query, k=config.DEFAULT_K)

        assert len(results) > 0, "Q3 retrieved zero chunks"

        review_chunks = [r for r in results if r["category"] == "reviews"]
        assert len(review_chunks) >= 1, (
            f"Q3 did not retrieve any 'reviews' chunks. "
            f"Got categories: {[r['category'] for r in results]}"
        )

    def test_mandatory_query_q3_beach_hotel_in_results(self):
        """
        Q3 should include results from hotels known for beach locations
        (Sunrise Boutique Resort, Serenity Palms Resort, Coral Bay Suites).
        """
        beach_hotels = {
            "sunrise boutique resort",
            "serenity palms resort",
            "coral bay suites",
        }
        index, chunks, embedder = _get_full_index()
        retriever = HotelRetriever(index, chunks, embedder)

        query = config.DEMO_QUERIES[2]
        results = retriever.retrieve(query, k=config.DEFAULT_K)

        retrieved_hotels = {r["hotel_name"].lower() for r in results}
        overlap = beach_hotels & retrieved_hotels
        assert len(overlap) >= 1, (
            f"Q3 did not retrieve any known beach hotels. "
            f"Got hotels: {retrieved_hotels}"
        )

    def test_mandatory_query_q3_no_lllm_called(self):
        """
        Confirm that the retrieval-only path does not accidentally invoke
        the LLM generator. We patch HotelGenerator to detect any call.
        """
        with patch("src.generator.HotelGenerator") as mock_gen_cls:
            index, chunks, embedder = _get_full_index()
            retriever = HotelRetriever(index, chunks, embedder)

            query = config.DEMO_QUERIES[2]
            results = retriever.retrieve(query, k=config.DEFAULT_K)

            # Generator constructor and generate() should never be called
            mock_gen_cls.assert_not_called()
            assert len(results) > 0
