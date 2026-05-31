"""
Hotel document preprocessor — cleaning and sentence-aware chunking.

Chunking justification:
    Sentence-aware chunking preserves semantic completeness of hotel
    policies and reviews. 200-token chunks balance specificity with
    context. 40-token overlap prevents boundary information loss.
"""
import re
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class HotelPreprocessor:
    """Cleans and chunks hotel documents into overlapping sentence-aware segments."""

    def clean_text(self, text: str) -> str:
        """
        Clean raw hotel document text.

        Args:
            text: Raw input string, may contain HTML, special chars, extra whitespace.

        Returns:
            Cleaned lowercase string with preserved sentence boundaries.

        Example:
            >>> p = HotelPreprocessor()
            >>> p.clean_text("<b>Hello  World!</b>")
            'hello world!'
        """
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        text = re.sub(r'\s([.,!?;:])', r'\1', text)
        return text

    def _tokenize_words(self, text: str) -> list:
        """Split text into word tokens by whitespace."""
        return text.split()

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences using punctuation boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc: dict) -> list:
        """
        Chunk a single hotel document into overlapping sentence-aware segments.

        Args:
            doc: Dict with keys: id, category, hotel_name, text.

        Returns:
            List of chunk dicts, each with:
                chunk_id, hotel_name, category, text, source_doc_id, token_count.

        Example:
            >>> p = HotelPreprocessor()
            >>> doc = {"id": "hotel_001", "hotel_name": "Test", "category": "description", "text": "Long text here."}
            >>> chunks = p.chunk_document(doc)
            >>> all("chunk_id" in c for c in chunks)
            True
        """
        cleaned = self.clean_text(doc["text"])
        sentences = self._split_into_sentences(cleaned)

        chunks = []
        current_tokens = []
        current_sentences = []
        chunk_index = 0

        for sentence in sentences:
            words = self._tokenize_words(sentence)
            if len(current_tokens) + len(words) > config.CHUNK_SIZE and current_tokens:
                chunk_text = " ".join(current_sentences)
                if len(current_tokens) >= config.MIN_CHUNK_TOKENS:
                    chunks.append({
                        "chunk_id": f"{doc['id']}_chunk_{chunk_index}",
                        "hotel_name": doc["hotel_name"],
                        "category": doc["category"],
                        "text": chunk_text,
                        "source_doc_id": doc["id"],
                        "token_count": len(current_tokens),
                    })
                    chunk_index += 1

                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_sentences):
                    w = self._tokenize_words(s)
                    if overlap_tokens + len(w) <= config.CHUNK_OVERLAP:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += len(w)
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = self._tokenize_words(" ".join(current_sentences))

            current_sentences.append(sentence)
            current_tokens.extend(words)

        if current_tokens and len(current_tokens) >= config.MIN_CHUNK_TOKENS:
            chunks.append({
                "chunk_id": f"{doc['id']}_chunk_{chunk_index}",
                "hotel_name": doc["hotel_name"],
                "category": doc["category"],
                "text": " ".join(current_sentences),
                "source_doc_id": doc["id"],
                "token_count": len(current_tokens),
            })

        if not chunks:
            full_text = self.clean_text(doc["text"])
            chunks.append({
                "chunk_id": f"{doc['id']}_chunk_0",
                "hotel_name": doc["hotel_name"],
                "category": doc["category"],
                "text": full_text,
                "source_doc_id": doc["id"],
                "token_count": len(self._tokenize_words(full_text)),
            })

        logger.debug("Document %s produced %d chunks", doc["id"], len(chunks))
        return chunks

    def process_all(self, documents: list) -> list:
        """
        Process all hotel documents into chunks.

        Args:
            documents: List of raw hotel document dicts.

        Returns:
            Flat list of all chunk dicts across all documents.

        Example:
            >>> p = HotelPreprocessor()
            >>> chunks = p.process_all(documents)
            >>> len(chunks) > 0
            True
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info("Processed %d documents into %d chunks", len(documents), len(all_chunks))
        return all_chunks
