"""
Hotel document preprocessor — cleaning and true semantic chunking.

Chunking strategy:
    Each sentence is embedded and windowed (SEMANTIC_WINDOW_SIZE sentences
    of context). Cosine similarity is computed between adjacent windows.
    Chunk boundaries are placed at similarity drops below the
    SEMANTIC_BREAKPOINT_PERCENTILE threshold — i.e. where the topic
    actually changes. A hard token cap (SEMANTIC_MAX_CHUNK_TOKENS) prevents
    oversized chunks when topics are long.

    Fallback: if no embedder is supplied (e.g. during testing), the
    preprocessor falls back to sentence-aware fixed-size chunking with
    CHUNK_SIZE / CHUNK_OVERLAP.
"""
import re
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class HotelPreprocessor:
    """Cleans hotel documents and chunks them using semantic boundary detection."""

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Clean raw hotel document text.

        Args:
            text: Raw string, may contain HTML, special chars, extra whitespace.

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
        return text.split()

    def _split_into_sentences(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # ------------------------------------------------------------------
    # Semantic chunking (primary path)
    # ------------------------------------------------------------------

    def _build_windows(self, sentences: list) -> list:
        """
        Wrap each sentence with SEMANTIC_WINDOW_SIZE neighbours for richer
        context when computing inter-sentence similarity.

        Args:
            sentences: List of sentence strings.

        Returns:
            List of windowed strings, one per sentence.
        """
        half = config.SEMANTIC_WINDOW_SIZE // 2
        windows = []
        for i in range(len(sentences)):
            start = max(0, i - half)
            end = min(len(sentences), i + half + 1)
            windows.append(" ".join(sentences[start:end]))
        return windows

    def _find_breakpoints(self, similarities: list) -> list:
        """
        Return sentence indices where a chunk boundary should be placed.

        A boundary is placed after sentence i when the similarity between
        window i and window i+1 falls below the SEMANTIC_BREAKPOINT_PERCENTILE
        of all similarities in the document.

        Args:
            similarities: List of floats, one per consecutive sentence pair.

        Returns:
            List of integer indices (into sentences list) where new chunks start.
        """
        if not similarities:
            return []
        threshold = float(np.percentile(similarities, config.SEMANTIC_BREAKPOINT_PERCENTILE))
        return [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    def _sentences_to_chunk(self, sentences: list, doc: dict, chunk_index: int) -> dict:
        """Build a chunk dict from a list of sentences."""
        text = " ".join(sentences)
        return {
            "chunk_id": f"{doc['id']}_chunk_{chunk_index}",
            "hotel_name": doc["hotel_name"],
            "category": doc["category"],
            "text": text,
            "source_doc_id": doc["id"],
            "token_count": len(self._tokenize_words(text)),
        }

    def _split_oversized(self, sentences: list, doc: dict, start_index: int) -> list:
        """
        Split a sentence group that exceeds SEMANTIC_MAX_CHUNK_TOKENS using
        the fixed-size fallback strategy.

        Args:
            sentences: Sentence list for one semantic segment.
            doc: Original document dict.
            start_index: Starting chunk_index counter.

        Returns:
            List of chunk dicts.
        """
        chunks = []
        current_sents = []
        current_tokens = 0
        idx = start_index

        for sent in sentences:
            n = len(self._tokenize_words(sent))
            if current_tokens + n > config.SEMANTIC_MAX_CHUNK_TOKENS and current_sents:
                if current_tokens >= config.MIN_CHUNK_TOKENS:
                    chunks.append(self._sentences_to_chunk(current_sents, doc, idx))
                    idx += 1
                current_sents = [sent]
                current_tokens = n
            else:
                current_sents.append(sent)
                current_tokens += n

        if current_sents and current_tokens >= config.MIN_CHUNK_TOKENS:
            chunks.append(self._sentences_to_chunk(current_sents, doc, idx))

        return chunks

    def semantic_chunk_document(self, doc: dict, embedder) -> list:
        """
        Chunk a document using semantic boundary detection.

        Embeds sentence windows, computes consecutive cosine similarities,
        places boundaries at the lowest-similarity gaps (percentile-based),
        then enforces a hard token cap on each resulting segment.

        Args:
            doc: Dict with keys id, hotel_name, category, text.
            embedder: HotelEmbedder instance with an encode-capable .model.

        Returns:
            List of chunk dicts with chunk_id, hotel_name, category, text,
            source_doc_id, token_count.

        Example:
            >>> chunks = p.semantic_chunk_document(doc, embedder)
            >>> all("chunk_id" in c for c in chunks)
            True
        """
        cleaned = self.clean_text(doc["text"])
        sentences = self._split_into_sentences(cleaned)

        if len(sentences) < 2:
            return [self._sentences_to_chunk(sentences or [cleaned], doc, 0)]

        windows = self._build_windows(sentences)
        embeddings = embedder.model.encode(windows, normalize_embeddings=True, show_progress_bar=False)

        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]
        breakpoints = self._find_breakpoints(similarities)

        segments = []
        prev = 0
        for bp in breakpoints:
            segments.append(sentences[prev:bp])
            prev = bp
        segments.append(sentences[prev:])
        segments = [s for s in segments if s]

        chunks = []
        idx = 0
        for seg in segments:
            token_count = len(self._tokenize_words(" ".join(seg)))
            if token_count <= config.SEMANTIC_MAX_CHUNK_TOKENS:
                if token_count >= config.MIN_CHUNK_TOKENS:
                    chunks.append(self._sentences_to_chunk(seg, doc, idx))
                    idx += 1
            else:
                sub = self._split_oversized(seg, doc, idx)
                chunks.extend(sub)
                idx += len(sub)

        if not chunks:
            chunks.append(self._sentences_to_chunk(sentences, doc, 0))

        logger.debug("Semantic chunked %s → %d chunks", doc["id"], len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Fixed-size fallback (used in tests / when no embedder is provided)
    # ------------------------------------------------------------------

    def chunk_document(self, doc: dict) -> list:
        """
        Sentence-aware fixed-size chunking fallback (no embedder required).

        Args:
            doc: Dict with keys id, category, hotel_name, text.

        Returns:
            List of chunk dicts.

        Example:
            >>> p = HotelPreprocessor()
            >>> chunks = p.chunk_document({"id": "hotel_001", "hotel_name": "Test", "category": "description", "text": "Some text."})
            >>> len(chunks) >= 1
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
                if len(current_tokens) >= config.MIN_CHUNK_TOKENS:
                    chunks.append(self._sentences_to_chunk(current_sentences, doc, chunk_index))
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
            chunks.append(self._sentences_to_chunk(current_sentences, doc, chunk_index))

        if not chunks:
            chunks.append(self._sentences_to_chunk([cleaned], doc, 0))

        return chunks

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_all(self, documents: list, embedder=None) -> list:
        """
        Process all hotel documents into chunks.

        Uses semantic chunking when embedder is provided, falls back to
        fixed-size sentence-aware chunking otherwise.

        Args:
            documents: List of raw hotel document dicts.
            embedder: Optional HotelEmbedder instance. If supplied, semantic
                      boundary detection is used. If None, falls back to
                      fixed-size chunking.

        Returns:
            Flat list of all chunk dicts across all documents.

        Example:
            >>> chunks = p.process_all(documents, embedder=embedder)
            >>> len(chunks) > len(documents)
            True
        """
        mode = "semantic" if embedder is not None else "fixed-size"
        logger.info("Chunking %d documents using %s strategy", len(documents), mode)

        all_chunks = []
        for doc in documents:
            if embedder is not None:
                chunks = self.semantic_chunk_document(doc, embedder)
            else:
                chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info("Produced %d chunks from %d documents (%s)", len(all_chunks), len(documents), mode)
        return all_chunks
