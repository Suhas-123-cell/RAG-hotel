# StayChat — Hotel Q&A RAG System

## 1. Project Overview

StayChat is a production-grade Retrieval-Augmented Generation (RAG) system
for answering hotel-related questions. It ingests a corpus of 40 synthetic
hotel documents covering five real-world categories — descriptions, amenities,
reviews, policies, and location — then chunks, embeds, and indexes them for
fast semantic retrieval. When a user asks a question, the system retrieves the
most relevant document chunks and passes them as grounded context to an LLM,
producing a factual answer with inline source citations.

The system is built entirely on free, open-source tools. Embeddings are
generated locally using `sentence-transformers/all-MiniLM-L6-v2`. Vector
search is powered by FAISS (CPU). Language generation uses Groq's free-tier
API (llama3-8b-8192). A Gradio web UI and an interactive CLI are both
included. Retrieval quality is evaluated with Precision@K and MRR, and
hallucination is controlled via strict prompting and a confidence gate.

---

## 2. Architecture Diagram

```
                        ┌─────────────────────────────────┐
                        │         hotel_documents.json     │
                        │         (40 hotel docs)          │
                        └────────────────┬────────────────┘
                                         │
                               HotelPreprocessor
                         (clean → sentence-aware chunk)
                                         │
                                    40 → ~90 chunks
                                         │
                               HotelEmbedder
                          (all-MiniLM-L6-v2, 384-dim)
                                         │
                                  FAISS IndexFlatIP
                                  (cosine similarity)
                                         │
                        ┌────────────────┴────────────────┐
                        │                                  │
                   User Query                        Confidence
                        │                              Gate
                  HotelRetriever                         │
              (top-k=5 by cosine sim)           [block if score<0.3]
                        │                                  │
                  Threshold filter                         │
                   (score >= 0.3)                          │
                        │                                  │
                 ┌──────┴──────┐                          │
                 │  top chunks  │◄────────────────────────┘
                 └──────┬──────┘
                        │
                 HotelGenerator
              (Groq llama3-8b-8192)
              strict context-only prompt
              inline [chunk_id] citations
                        │
                   Final Answer
                        │
               ┌────────┴────────┐
               │                 │
           CLI (main.py)   Gradio UI (app.py)
```

---

## 3. Tech Stack

| Component | Library / Service | Version | Role |
|-----------|-------------------|---------|------|
| Embeddings | sentence-transformers | 3.0.1 | Local text embedding (free) |
| Vector store | faiss-cpu | 1.8.0 | Nearest-neighbour search |
| LLM | Groq API (llama3-8b-8192) | groq 0.9.0 | Answer generation (free tier) |
| Orchestration | LangChain | 0.2.16 | Pipeline utilities |
| Web UI | Gradio | 4.44.0 | Browser demo interface |
| Data handling | Pandas | 2.2.3 | Dataset utilities |
| Text processing | NLTK | 3.9.1 | Preprocessing support |
| Env management | python-dotenv | 1.0.1 | API key loading |
| Testing | pytest | 8.3.4 | Unit and integration tests |

---

## 4. Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/Suhas-123-cell/RAG-hotel.git
cd RAG-hotel

# 2. Create a virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env and replace "your_key_here" with your actual key:
# GROQ_API_KEY=gsk_...

# 5. Run the full system
python main.py
```

---

## 5. Running the System

```bash
# Run everything (demo + eval + hallucination demo)
python main.py

# Run only the 3 mandatory demo queries
python main.py --mode demo

# Run the evaluation suite (Precision@K, MRR, qualitative analysis)
python main.py --mode eval

# Interactive CLI chat
python main.py --mode chat

# Hallucination control demonstration
python main.py --mode hallucination

# Force rebuild of FAISS index (if dataset changes)
python main.py --rebuild

# Launch Gradio web UI (opens at http://localhost:7860)
python app.py

# Run tests
pytest tests/ -v
pytest tests/ -v -m "not slow"   # fast tests only (no model needed)
```

---

## 6. Dataset Description

`data/hotel_documents.json` contains **40 documents** across **5 hotels**:

| Hotel | Documents |
|-------|-----------|
| The Azure Grand | 8 |
| Sunrise Boutique Resort | 8 |
| Coral Bay Suites | 8 |
| The Pinnacle Hotel | 8 |
| Serenity Palms Resort | 8 |

| Category | Count | Content |
|----------|-------|---------|
| description | 8 | Hotel overview, atmosphere, history |
| amenities | 8 | Facilities, services, inclusions (WiFi, breakfast, spa, pool) |
| reviews | 12 | Guest reviews — positive, neutral, and negative |
| policies | 6 | Cancellation, check-in/out, pet, smoking policies |
| location | 6 | Nearby attractions, transport, geographic context |

---

## 7. Chunking Strategy and Justification

**Strategy**: True semantic chunking via embedding similarity (with fixed-size fallback)

**How it works**:
1. Each sentence is embedded using all-MiniLM-L6-v2 within a `SEMANTIC_WINDOW_SIZE=3`
   sentence context window to capture local discourse
2. Cosine similarity is computed between adjacent sentence windows
3. Chunk boundaries are placed at the bottom `SEMANTIC_BREAKPOINT_PERCENTILE=25`
   of similarities — i.e. where topics genuinely shift
4. A hard cap of `SEMANTIC_MAX_CHUNK_TOKENS=300` splits any oversized semantic chunks

**Why this beats fixed-size chunking**:

| Scenario | Fixed-size | Semantic |
|----------|-----------|---------|
| Long cancellation policy | May cut mid-clause | Keeps whole policy clause together |
| Review with topic shift | Mixed sentiment in one chunk | Splits at positive/negative boundary |
| Short factual doc | Same | Same |

**Fallback**: When no embedder is available (e.g. unit tests), the preprocessor
falls back to sentence-aware fixed-size chunking (`CHUNK_SIZE=200`, `CHUNK_OVERLAP=40`).

---

## 8. Embedding Model Rationale

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

- **Free and local**: Runs on CPU, no API cost or latency overhead
- **384-dim vectors**: Compact enough for fast FAISS search even on large datasets
- **Strong semantic matching**: Trained on 1B+ sentence pairs; captures
  synonyms and paraphrases (e.g., "free WiFi" ~ "complimentary internet access")
- **Speed**: Encodes ~14,000 sentences/second on CPU
- **FAISS compatibility**: Outputs L2-normalised vectors, making inner-product
  search equivalent to cosine similarity (IndexFlatIP)

---

## 9. Retrieval Design — Hybrid BM25 + Dense + RRF

**Why hybrid**: Dense-only retrieval misses exact keyword matches (hotel names,
room numbers, policy terms). BM25-only misses paraphrases and synonyms.
Hybrid combines both signals without needing score normalisation.

**Pipeline**:
```
query
 ├── FAISS dense search  → top-20 by cosine similarity (semantic matches)
 ├── BM25 sparse search  → top-20 by TF-IDF term score (keyword matches)
 └── RRF merge           → score = Σ 1/(RRF_K=60 + rank_i)  → top-5 final
```

**Key parameters** (`config.py`):
- `DEFAULT_K = 5` — final results after RRF merge
- `HYBRID_CANDIDATE_K = 20` — candidates fetched from each system before merge
- `RRF_K = 60` — RRF constant; higher value flattens rank differences
- `BM25_K1 = 1.5`, `BM25_B = 0.75` — standard Okapi BM25 tuning values
- `SIMILARITY_THRESHOLD = 0.3` — post-RRF floor to drop near-zero contributors

---

## 10. Hallucination Control Methods

### Method 1 — Prompt Injection Guard (input + output)

Every query is scanned against `INJECTION_PATTERNS` in `config.py` before
reaching the LLM. Patterns include:

- `ignore (all) previous instructions`
- `you are now / act as / pretend to be`
- `reveal your system prompt`
- `override / jailbreak`
- Raw XML tags (`<system>`, `<user>`) injected in the query

Flagged queries are rejected with an error message — the LLM is never called.
The output is also validated to catch edge cases where the model echoed
injected instructions through.

### Method 2 — XML-Delimited Prompt Structure

Context and the user question are wrapped in distinct XML tags:

```
<context>
[chunk_id] (hotel — category)
...chunk text...
</context>

<question>user query here</question>
```

The system prompt explicitly instructs the model to ignore anything inside
`<question>` that tries to change its behaviour. XML delimiters make the
boundary structurally unambiguous, not just semantically instructed.

### Method 3 — Strict Context-Only System Prompt

The LLM is instructed to answer ONLY from `<context>`. If the answer is not
there, it must respond with a fixed refusal string. Every fact must cite a
`[chunk_id]`. This removes fallback to parametric memory.

### Method 4 — Confidence Gate

If the highest retrieval score after RRF is below `1/(RRF_K * 2)`, generation
is blocked entirely:

> "Insufficient context confidence. Cannot answer reliably."

Catches out-of-domain queries (e.g., stock prices, general trivia) before
they reach the LLM with weakly-related hotel context.

---

## 11. Evaluation Results

### Precision@K (k=5)

**Formula**: `P@k = |relevant intersect retrieved[:k]| / k`

| Query | Relevant in top-5 | Precision@5 | Working |
|-------|------------------|-------------|---------|
| Q1 — WiFi & Breakfast | 5/5 | **1.00** | 5 amenity chunks for 4 hotels / 5 = 1.00 |
| Q2 — Coral Bay Policy | 2/5 | **0.40** | 2 Coral Bay policy chunks / 5 = 0.40 |
| Q3 — Beach + Reviews | 5/5 | **1.00** | 5 beach hotel review/location chunks / 5 = 1.00 |
| **Mean** | | **0.80** | |

### MRR (Mean Reciprocal Rank)

**Formula**: `MRR = 1 / rank_of_first_relevant_result`

| Query | First relevant rank | MRR |
|-------|---------------------|-----|
| Q1 | 1 | **1.00** |
| Q2 | 1 | **1.00** |
| Q3 | 1 | **1.00** |
| **Mean** | | **1.00** |

---

## 12. Sample Outputs

Full pre-generated outputs with chunk tables and faithfulness verdicts are in
[`outputs/sample_outputs.md`](outputs/sample_outputs.md).

**Quick preview — Q2 answer**:

> The cancellation policy for **Coral Bay Suites** is as follows:
> - Free cancellation up to 7 days before check-in. [hotel_026_chunk_0]
> - Cancellations within 48 hours incur one night's room rate. [hotel_026_chunk_1]
> - Cancellations 2-7 days prior incur a 50% charge. [hotel_026_chunk_0]

---

## 13. Known Limitations

- **40-document dataset**: Corpus is synthetic. Real-world performance depends
  on quality and coverage of actual hotel content.
- **Static ground truth**: `EVAL_GROUND_TRUTH` doc IDs in `config.py` are
  hand-coded; must be updated if the dataset changes.
- **Groq free-tier rate limits**: ~30 requests/minute. The `--mode all` run
  makes 6-8 LLM calls; a `groq.RateLimitError` may appear under burst load.
- **No query rewriting**: Multi-hop questions are not decomposed; retrieval
  relies on a single embedding pass.
- **English only**: Both the embedding model and LLM perform best on English.

---

## 14. Future Improvements

- **Hybrid retrieval**: Combine BM25 keyword search with dense embeddings
  (Reciprocal Rank Fusion) to improve recall for exact-match queries.
- **Query rewriting**: Use the LLM to expand ambiguous queries before
  retrieval (HyDE — Hypothetical Document Embeddings).
- **Re-ranking**: Add a cross-encoder re-ranker after FAISS retrieval to
  improve precision from k=20 down to k=5.
- **Real dataset**: Scrape or license actual hotel content from booking
  platforms and integrate with a property management system.
- **Streaming responses**: Add SSE/WebSocket streaming to the Gradio UI.
- **Conversation memory**: Implement multi-turn chat with LangChain
  `ConversationBufferWindowMemory` so follow-up questions retain context.
