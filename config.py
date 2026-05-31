"""Central configuration for StayChat RAG system."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"
INDEX_DIR = BASE_DIR / "index"

# Dataset
DOCUMENTS_FILE = DATA_DIR / "hotel_documents.json"

# Chunking
CHUNK_SIZE = 200        # tokens
CHUNK_OVERLAP = 40      # tokens
MIN_CHUNK_TOKENS = 30   # drop chunks smaller than this

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# FAISS index paths
FAISS_INDEX_PATH = INDEX_DIR / "hotel_faiss.index"
CHUNKS_PICKLE_PATH = INDEX_DIR / "hotel_chunks.pkl"

# Retrieval
DEFAULT_K = 5           # top-k chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # min cosine similarity score

# LLM
GROQ_MODEL = "llama3-8b-8192"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
MAX_TOKENS = 1024
TEMPERATURE = 0.1       # low for faithfulness

# Gradio
GRADIO_PORT = 7860
GRADIO_TITLE = "StayChat Hotel Q&A — RAG Demo"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

# Evaluation — ground truth chunk IDs for each mandatory query
# These are manually set after building the dataset; update after indexing
EVAL_GROUND_TRUTH = {
    "Which hotels have free WiFi and complimentary breakfast?": [],  # filled at runtime
    "What is the cancellation policy of Coral Bay Suites?": [],
    "Suggest a hotel with excellent reviews near the beach.": [],
}

# Mandatory demo queries
DEMO_QUERIES = [
    "Which hotels have free WiFi and complimentary breakfast?",
    "What is the cancellation policy of Coral Bay Suites?",
    "Suggest a hotel with excellent reviews near the beach.",
]
