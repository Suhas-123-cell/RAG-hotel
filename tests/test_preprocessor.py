"""
Tests for HotelPreprocessor — cleaning and chunking logic.

Runs without any external services (no GPU, no API key required).
"""
import json
import sys
from pathlib import Path

import pytest

# Make sure the project root is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.preprocessor import HotelPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preprocessor():
    return HotelPreprocessor()


@pytest.fixture(scope="module")
def hotel_documents():
    """Load the real hotel_documents.json dataset."""
    with open(config.DOCUMENTS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def sample_doc():
    """A minimal hotel document for unit tests that don't need the full corpus."""
    return {
        "id": "test_001",
        "hotel_name": "Test Hotel",
        "category": "description",
        "text": (
            "The Grand Ocean Hotel is a five-star beachfront resort. "
            "It offers 200 rooms with ocean views. "
            "The spa is open daily from 8am to 10pm. "
            "Guests enjoy complimentary WiFi and free breakfast every morning. "
            "The outdoor pool has a dedicated lifeguard on duty. "
            "Cancellations must be made 48 hours in advance to avoid charges. "
            "The hotel is located just 5 minutes from the city centre. "
            "Room service is available 24 hours a day, seven days a week. "
            "The hotel features a rooftop bar with panoramic sunset views. "
            "Conference facilities accommodate up to 500 guests. "
        ),
    }


@pytest.fixture
def long_doc():
    """A document long enough to guarantee multiple chunks at CHUNK_SIZE=200."""
    sentences = [
        f"Sentence number {i} describes a unique feature of this luxury hotel property. "
        f"The amenity in question is highly rated by international travel publications."
        for i in range(1, 60)
    ]
    return {
        "id": "test_long_001",
        "hotel_name": "Long Test Hotel",
        "category": "amenities",
        "text": " ".join(sentences),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_clean_text_removes_html(self, preprocessor):
        """HTML tags must be stripped; surrounding words must survive."""
        raw = "<b>Welcome</b> to the <em>Grand Hotel</em>!"
        result = preprocessor.clean_text(raw)
        assert "<b>" not in result
        assert "<em>" not in result
        assert "welcome" in result
        assert "grand hotel" in result

    def test_clean_text_removes_html_attributes(self, preprocessor):
        """Tags with attributes (href, class, style…) must also be removed."""
        raw = '<a href="https://example.com" class="link">Book now</a>'
        result = preprocessor.clean_text(raw)
        assert "<a" not in result
        assert "href" not in result
        assert "book now" in result

    def test_clean_text_collapses_whitespace(self, preprocessor):
        """Multiple consecutive spaces / newlines should become a single space."""
        raw = "Free   WiFi\n\nand\t\tbreakfast."
        result = preprocessor.clean_text(raw)
        assert "  " not in result
        assert "\n" not in result
        assert "\t" not in result

    def test_clean_text_lowercases(self, preprocessor):
        """Output must be fully lowercase."""
        raw = "The AZURE Grand Hotel OFFERS Free WiFi."
        result = preprocessor.clean_text(raw)
        assert result == result.lower()

    def test_clean_text_preserves_sentence_punctuation(self, preprocessor):
        """Periods, commas, exclamation and question marks must be retained."""
        raw = "Great location! Is WiFi free? Yes, it is."
        result = preprocessor.clean_text(raw)
        assert "!" in result
        assert "?" in result
        assert "." in result
        assert "," in result

    def test_clean_text_empty_string(self, preprocessor):
        """Empty input must return an empty string without raising."""
        assert preprocessor.clean_text("") == ""

    def test_clean_text_only_html(self, preprocessor):
        """Input consisting solely of tags must produce an empty or whitespace-only result."""
        result = preprocessor.clean_text("<div><p><br/></p></div>")
        assert result.strip() == ""


class TestChunkSize:
    def test_chunk_size_within_bounds(self, preprocessor, long_doc):
        """No chunk should exceed CHUNK_SIZE * 1.5 tokens (word-split approximation)."""
        max_allowed = int(config.CHUNK_SIZE * 1.5)
        chunks = preprocessor.chunk_document(long_doc)
        assert len(chunks) > 0, "Expected at least one chunk from a long document"
        for chunk in chunks:
            token_count = len(chunk["text"].split())
            assert token_count <= max_allowed, (
                f"Chunk {chunk['chunk_id']} has {token_count} tokens, "
                f"exceeds allowed max of {max_allowed}"
            )

    def test_chunk_size_respects_min_tokens(self, preprocessor, long_doc):
        """Every chunk must have at least MIN_CHUNK_TOKENS tokens."""
        chunks = preprocessor.chunk_document(long_doc)
        for chunk in chunks:
            token_count = len(chunk["text"].split())
            assert token_count >= config.MIN_CHUNK_TOKENS, (
                f"Chunk {chunk['chunk_id']} has only {token_count} tokens, "
                f"below minimum of {config.MIN_CHUNK_TOKENS}"
            )

    def test_stored_token_count_matches_text(self, preprocessor, sample_doc):
        """The stored token_count field should equal the word count of the text."""
        chunks = preprocessor.chunk_document(sample_doc)
        for chunk in chunks:
            actual = len(chunk["text"].split())
            assert chunk["token_count"] == actual, (
                f"Stored token_count {chunk['token_count']} != actual word count {actual}"
            )


class TestOverlap:
    def test_overlap_preserved(self, preprocessor, long_doc):
        """
        Consecutive chunks must share some words (evidence of overlap).
        We check that the last sentence of chunk[i] appears in chunk[i+1].
        """
        chunks = preprocessor.chunk_document(long_doc)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks produced to test overlap")

        found_overlap = False
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i]["text"].split())
            words_b = set(chunks[i + 1]["text"].split())
            shared = words_a & words_b
            if shared:
                found_overlap = True
                break

        assert found_overlap, (
            "No consecutive chunk pair shares any words — overlap appears absent"
        )

    def test_overlap_not_full_duplicate(self, preprocessor, long_doc):
        """Consecutive chunks must NOT be identical — they should differ in content."""
        chunks = preprocessor.chunk_document(long_doc)
        for i in range(len(chunks) - 1):
            assert chunks[i]["text"] != chunks[i + 1]["text"], (
                f"Chunks {i} and {i+1} are identical — possible full-duplicate overlap"
            )


class TestAllDocsProduceChunks:
    def test_all_docs_produce_chunks(self, preprocessor, hotel_documents):
        """Every document in hotel_documents.json must produce at least one chunk."""
        for doc in hotel_documents:
            chunks = preprocessor.chunk_document(doc)
            assert len(chunks) >= 1, (
                f"Document {doc['id']} ({doc['hotel_name']}) produced zero chunks"
            )

    def test_chunk_ids_are_unique(self, preprocessor, hotel_documents):
        """Chunk IDs across the entire corpus must be globally unique."""
        all_chunks = preprocessor.process_all(hotel_documents)
        ids = [c["chunk_id"] for c in all_chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected in corpus"

    def test_chunks_inherit_metadata(self, preprocessor, hotel_documents):
        """Every chunk must carry hotel_name, category, and source_doc_id."""
        all_chunks = preprocessor.process_all(hotel_documents)
        for chunk in all_chunks:
            assert chunk["hotel_name"], f"Missing hotel_name in {chunk['chunk_id']}"
            assert chunk["category"], f"Missing category in {chunk['chunk_id']}"
            assert chunk["source_doc_id"], f"Missing source_doc_id in {chunk['chunk_id']}"

    def test_total_corpus_chunk_count_reasonable(self, preprocessor, hotel_documents):
        """Sanity check: 40 docs should produce at least 40 and at most 2000 chunks."""
        all_chunks = preprocessor.process_all(hotel_documents)
        assert len(all_chunks) >= 40, "Fewer chunks than documents — something is wrong"
        assert len(all_chunks) <= 2000, "Unexpectedly large number of chunks"
