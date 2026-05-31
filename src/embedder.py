"""
Hotel document embedder — sentence-transformers + FAISS index management.
"""
import logging
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class HotelEmbedder:
    """Embeds hotel chunks using all-MiniLM-L6-v2 and manages the FAISS index."""

    def __init__(self):
        """Initialize the sentence-transformer embedding model."""
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")

    def embed_chunks(self, chunks: list) -> np.ndarray:
        """Embed a list of chunk dicts. Returns float32 array of shape (n_chunks, EMBEDDING_DIM)."""
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("Embedding complete. Shape: %s", embeddings.shape)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns float32 array of shape (1, EMBEDDING_DIM)."""
        vec = self.model.encode([query], normalize_embeddings=True)
        return vec.astype(np.float32)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS IndexFlatIP (inner product = cosine for normalized vectors)."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info("FAISS index built with %d vectors (dim=%d)", index.ntotal, dim)
        return index

    def save_index(self, index: faiss.Index, chunks: list) -> None:
        """Save FAISS index and chunks list to FAISS_INDEX_PATH and CHUNKS_PICKLE_PATH."""
        config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))
        with open(config.CHUNKS_PICKLE_PATH, "wb") as f:
            pickle.dump(chunks, f)
        logger.info("Index saved to %s", config.FAISS_INDEX_PATH)
        logger.info("Chunks saved to %s", config.CHUNKS_PICKLE_PATH)

    def load_index(self) -> tuple:
        """Load FAISS index and chunks list from disk. Raises FileNotFoundError if missing."""
        if not config.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {config.FAISS_INDEX_PATH}")
        if not config.CHUNKS_PICKLE_PATH.exists():
            raise FileNotFoundError(f"Chunks file not found at {config.CHUNKS_PICKLE_PATH}")

        index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        with open(config.CHUNKS_PICKLE_PATH, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Index loaded: %d vectors", index.ntotal)
        logger.info("Chunks loaded: %d chunks", len(chunks))
        return index, chunks
