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

# Chunking — sentence-aware fallback (used when embedder not available)
CHUNK_SIZE = 200        # tokens
CHUNK_OVERLAP = 40      # tokens
MIN_CHUNK_TOKENS = 30   # drop chunks smaller than this

# Semantic chunking (true topic-boundary detection via embedding similarity)
SEMANTIC_BREAKPOINT_PERCENTILE = 25   # bottom 25% similarity scores → chunk boundaries
SEMANTIC_WINDOW_SIZE = 3              # sentences either side when embedding for context
SEMANTIC_MAX_CHUNK_TOKENS = 300       # hard cap — splits oversized semantic chunks

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# FAISS index paths
FAISS_INDEX_PATH = INDEX_DIR / "hotel_faiss.index"
CHUNKS_PICKLE_PATH = INDEX_DIR / "hotel_chunks.pkl"

# BM25 sparse retrieval
BM25_K1 = 1.5
BM25_B = 0.75

# Hybrid retrieval (BM25 + Dense → RRF)
RRF_K = 60              # RRF constant; higher = less rank-gap impact
HYBRID_CANDIDATE_K = 50 # candidates fetched from each system before merge

# Retrieval
DEFAULT_K = 5           # final top-k after RRF merge
GENERATION_CONTEXT_K = 14  # broader hidden context for answer generation
SIMILARITY_THRESHOLD = 0.3  # min cosine similarity score (dense gate)

# Prompt injection guard patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(you\s+are|your\s+role|everything)",
    r"you\s+are\s+now|act\s+as\s+|pretend\s+(to\s+be|you\s+are)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
    r"disregard\s+(the\s+)?context",
    r"override|jailbreak",
    r"do\s+not\s+(follow|use)\s+(the\s+)?context",
    r"new\s+instruction|updated\s+instruction",
    r"</?system>|</?user>|</?assistant>",
    # "ignore all (the) system commands/prompts/rules/constraints"
    r"ignore\s+(all\s+)?(the\s+)?(system\s+)?(commands?|prompts?|rules?|constraints?)",
    # broader: "ignore (all) (the) instructions/commands/rules" without requiring "system"
    r"ignore\s+(all\s+)?(the\s+)(instructions?|commands?|rules?)",
    # data exfiltration phrasing: "give me all the information you have"
    r"give\s+me\s+(all\s+)?(the\s+)?information\s+(you\s+have|that\s+you\s+have)",
    # bypass attempts
    r"bypass\s+(the\s+)?(system|filter|restriction|guard)",
    # classic DAN jailbreak
    r"\bDAN\b|do\s+anything\s+now",
    # role injection via plain text labels
    r"(?:system|user|assistant)\s*:\s",
]

# LLM
GROQ_MODEL = "llama-3.1-8b-instant"
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
