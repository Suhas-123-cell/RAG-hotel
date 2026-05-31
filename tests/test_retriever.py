"""
Tests for HotelRetriever — semantic search and threshold filtering.

All tests mock the FAISS index and embedder so no GPU or model download
is required. Tests run fully in-process with no external dependencies.

The sentence_transformers and groq packages may not be installed in the
test environment; they are mocked at the sys.modules level before the
project modules are imported so that the heavy model is never loaded.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies before any project module is imported.
# This lets test_retriever.py run even when sentence_transformers / groq are
# not installed in the active Python environment.
# ---------------------------------------------------------------------------
for _mod in (
    "sentence_transformers",
    "groq",
    "dotenv",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# dotenv sub-import used by generator.py
if "dotenv" not in sys.modules or not hasattr(sys.modules["dotenv"], "load_dotenv"):
    _mock_dotenv = MagicMock()
    _mock_dotenv.load_dotenv = MagicMock()
    sys.modules["dotenv"] = _mock_dotenv

import config
from src.retriever import HotelRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list:
    """Create n fake chunk dicts with deterministic content."""
    return [
        {
            "chunk_id": f"hotel_00{i+1}_chunk_0",
            "hotel_name": f"Hotel {i+1}",
            "category": "amenities",
            "text": f"This is chunk number {i+1} about hotel amenities and services.",
            "source_doc_id": f"hotel_00{i+1}",
            "token_count": 10,
        }
        for i in range(n)
    ]


def _make_mock_embedder(query_vec: np.ndarray) -> MagicMock:
    """Return a mock HotelEmbedder whose embed_query always returns query_vec."""
    embedder = MagicMock()
    embedder.embed_query.return_value = query_vec
    return embedder


def _make_mock_index(scores: list, indices: list) -> MagicMock:
    """
    Return a mock FAISS index whose search() returns the given scores/indices.

    FAISS search returns 2-D arrays: shape (1, k).
    """
    index = MagicMock()
    index.search.return_value = (
        np.array([scores], dtype=np.float32),
        np.array([indices], dtype=np.int64),
    )
    return index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_query_vec():
    """Dummy query embedding — shape (1, EMBEDDING_DIM)."""
    vec = np.random.rand(1, config.EMBEDDING_DIM).astype(np.float32)
    return vec


@pytest.fixture
def five_chunks():
    return _make_chunks(5)


@pytest.fixture
def ten_chunks():
    return _make_chunks(10)


# ---------------------------------------------------------------------------
# test_retrieval_returns_k_results
# ---------------------------------------------------------------------------

class TestRetrievalReturnsKResults:
    def test_retrieval_returns_k_results_default(self, fake_query_vec, five_chunks):
        """retrieve() with default k must return exactly DEFAULT_K results (if available)."""
        k = config.DEFAULT_K
        scores = [0.9 - 0.1 * i for i in range(k)]
        indices = list(range(k))

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, five_chunks, mock_embedder)

        results = retriever.retrieve("What amenities are available?")

        assert len(results) == k, (
            f"Expected {k} results, got {len(results)}"
        )
        mock_embedder.embed_query.assert_called_once_with("What amenities are available?")
        mock_index.search.assert_called_once()

    def test_retrieval_returns_k_results_custom(self, fake_query_vec, ten_chunks):
        """retrieve(k=3) must return exactly 3 results."""
        k = 3
        scores = [0.85, 0.75, 0.65]
        indices = [0, 1, 2]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, ten_chunks, mock_embedder)

        results = retriever.retrieve("Free WiFi?", k=k)

        assert len(results) == k

    def test_retrieval_skips_minus_one_indices(self, fake_query_vec, five_chunks):
        """
        FAISS returns -1 for unfilled slots when the index has fewer vectors than k.
        retrieve() must silently skip -1 indices.
        """
        scores = [0.9, 0.8, -1.0, -1.0, -1.0]
        indices = [0, 1, -1, -1, -1]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, five_chunks, mock_embedder)

        results = retriever.retrieve("Any query", k=5)

        # Only 2 valid indices, -1s should be dropped
        assert len(results) == 2
        for r in results:
            assert r.get("chunk_id") is not None

    def test_retrieval_returns_empty_on_all_minus_one(self, fake_query_vec, five_chunks):
        """If every FAISS slot is -1, retrieve() must return an empty list."""
        scores = [-1.0, -1.0, -1.0]
        indices = [-1, -1, -1]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, five_chunks, mock_embedder)

        results = retriever.retrieve("Empty query", k=3)
        assert results == []


# ---------------------------------------------------------------------------
# test_scores_are_sorted_descending
# ---------------------------------------------------------------------------

class TestScoresSortedDescending:
    def test_scores_are_sorted_descending(self, fake_query_vec, ten_chunks):
        """Results from retrieve() must be sorted by score in descending order."""
        # Deliberately provide out-of-order scores to verify sorting
        scores = [0.5, 0.9, 0.3, 0.7, 0.1]
        indices = [0, 1, 2, 3, 4]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, ten_chunks, mock_embedder)

        results = retriever.retrieve("Hotel breakfast?", k=5)

        returned_scores = [r["score"] for r in results]
        assert returned_scores == sorted(returned_scores, reverse=True), (
            f"Scores not descending: {returned_scores}"
        )

    def test_scores_are_sorted_descending_already_sorted(self, fake_query_vec, five_chunks):
        """retrieve() stays correct even when FAISS already returns scores in order."""
        scores = [0.95, 0.88, 0.72, 0.60, 0.45]
        indices = [0, 1, 2, 3, 4]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, five_chunks, mock_embedder)

        results = retriever.retrieve("Cancellation policy?", k=5)

        returned_scores = [r["score"] for r in results]
        assert returned_scores == sorted(returned_scores, reverse=True)

    def test_first_result_has_highest_score(self, fake_query_vec, five_chunks):
        """The first result must always have the maximum score in the result set."""
        scores = [0.4, 0.99, 0.6, 0.2, 0.75]
        indices = [0, 1, 2, 3, 4]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, five_chunks, mock_embedder)

        results = retriever.retrieve("Beach hotel?", k=5)

        assert results[0]["score"] == max(r["score"] for r in results)

    def test_score_stored_as_float(self, fake_query_vec, five_chunks):
        """Scores on returned chunks must be Python floats, not numpy types."""
        scores = [0.8, 0.7, 0.6]
        indices = [0, 1, 2]

        mock_index = _make_mock_index(scores, indices)
        mock_embedder = _make_mock_embedder(fake_query_vec)
        retriever = HotelRetriever(mock_index, five_chunks, mock_embedder)

        results = retriever.retrieve("Test", k=3)
        for r in results:
            assert isinstance(r["score"], float), (
                f"score type is {type(r['score'])}, expected float"
            )


# ---------------------------------------------------------------------------
# test_threshold_filter_removes_low_scores
# ---------------------------------------------------------------------------

class TestThresholdFilter:
    def test_threshold_filter_removes_low_scores(self, fake_query_vec, ten_chunks):
        """filter_by_threshold must drop all chunks whose score is below min_score."""
        threshold = 0.5
        chunks_with_scores = []
        raw_scores = [0.9, 0.7, 0.4, 0.6, 0.2]
        for i, score in enumerate(raw_scores):
            c = dict(ten_chunks[i])
            c["score"] = score
            chunks_with_scores.append(c)

        mock_index = MagicMock()
        mock_embedder = MagicMock()
        retriever = HotelRetriever(mock_index, ten_chunks, mock_embedder)

        filtered = retriever.filter_by_threshold(chunks_with_scores, min_score=threshold)

        assert all(r["score"] >= threshold for r in filtered), (
            f"Some results have score < {threshold}: "
            f"{[r['score'] for r in filtered]}"
        )
        # 0.9, 0.7, 0.6 pass; 0.4 and 0.2 are dropped
        assert len(filtered) == 3

    def test_threshold_filter_keeps_all_above_threshold(self, ten_chunks):
        """When all scores are above the threshold, nothing should be dropped."""
        threshold = 0.1
        chunks_with_scores = []
        for i, score in enumerate([0.95, 0.88, 0.77, 0.66]):
            c = dict(ten_chunks[i])
            c["score"] = score
            chunks_with_scores.append(c)

        retriever = HotelRetriever(MagicMock(), ten_chunks, MagicMock())
        filtered = retriever.filter_by_threshold(chunks_with_scores, min_score=threshold)

        assert len(filtered) == len(chunks_with_scores)

    def test_threshold_filter_removes_all_below_threshold(self, ten_chunks):
        """When all scores are below the threshold, an empty list is returned."""
        threshold = 0.9
        chunks_with_scores = []
        for i, score in enumerate([0.1, 0.2, 0.3]):
            c = dict(ten_chunks[i])
            c["score"] = score
            chunks_with_scores.append(c)

        retriever = HotelRetriever(MagicMock(), ten_chunks, MagicMock())
        filtered = retriever.filter_by_threshold(chunks_with_scores, min_score=threshold)

        assert filtered == []

    def test_threshold_filter_uses_default_from_config(self, ten_chunks):
        """When min_score is omitted, the config.SIMILARITY_THRESHOLD is used."""
        threshold = config.SIMILARITY_THRESHOLD
        chunks_with_scores = []
        for i, score in enumerate([threshold + 0.1, threshold - 0.1, threshold + 0.2]):
            c = dict(ten_chunks[i])
            c["score"] = score
            chunks_with_scores.append(c)

        retriever = HotelRetriever(MagicMock(), ten_chunks, MagicMock())
        filtered = retriever.filter_by_threshold(chunks_with_scores)

        assert all(r["score"] >= threshold for r in filtered)
        assert len(filtered) == 2  # the one below threshold is dropped

    def test_threshold_filter_boundary_exact_value(self, ten_chunks):
        """A score exactly equal to min_score should be KEPT (>= not >)."""
        threshold = 0.5
        c = dict(ten_chunks[0])
        c["score"] = threshold  # exactly at boundary

        retriever = HotelRetriever(MagicMock(), ten_chunks, MagicMock())
        filtered = retriever.filter_by_threshold([c], min_score=threshold)

        assert len(filtered) == 1, (
            "Chunk with score exactly at threshold should be kept"
        )

    def test_threshold_filter_preserves_original_chunk_data(self, ten_chunks):
        """filter_by_threshold must not mutate the chunk dicts that it keeps."""
        c = dict(ten_chunks[0])
        c["score"] = 0.8
        original_text = c["text"]

        retriever = HotelRetriever(MagicMock(), ten_chunks, MagicMock())
        filtered = retriever.filter_by_threshold([c], min_score=0.5)

        assert filtered[0]["text"] == original_text
        assert filtered[0]["hotel_name"] == c["hotel_name"]
