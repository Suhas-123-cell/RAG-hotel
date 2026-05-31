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

**Strategy**: Sentence-aware sliding window chunking

**Parameters** (from `config.py`):
- `CHUNK_SIZE = 200` tokens
- `CHUNK_OVERLAP = 40` tokens
- `MIN_CHUNK_TOKENS = 30` (drops micro-fragments)

**Justification**:
- *Sentence-aware*: Splitting at sentence boundaries preserves the semantic
  completeness of policy clauses and review opinions. Word-level splitting
  would break mid-clause, losing the predicate that makes a policy meaningful.
- *200 tokens*: Long enough to contain a complete fact (e.g., cancellation
  terms) without drowning retrieval in noise. Short enough that multiple
  relevant chunks can fit in the LLM context window.
- *40-token overlap*: Prevents information loss at chunk boundaries. A policy
  clause that spans two sentences (common in hotel T&Cs) is fully captured by
  at least one chunk.

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

## 9. Retrieval Design

**k = 5**: Chosen to balance recall (enough context for multi-hotel questions)
against context window length (5 x ~200 tokens = ~1,000 tokens of context,
well within the 8K context of llama3-8b-8192).

**Similarity threshold = 0.3**: The cosine similarity distribution for
off-topic queries (e.g., stock prices, general trivia) peaks below 0.3 when
tested against hotel documents. Setting the threshold here catches
out-of-domain queries before they reach the LLM, without over-filtering
legitimate low-similarity but relevant chunks (e.g., amenities asked about
via indirect phrasing).

---

## 10. Hallucination Control Methods

### Method 1 — Strict Context-Only Prompting

The system prompt instructs the LLM:

> "Answer ONLY using the provided context. If the answer is not in the
> context, respond exactly with: 'I don't have enough information to answer
> this from the available hotel data.' For every fact you state, cite the
> source document ID in brackets like [hotel_003_chunk_2]."

This removes the model's fallback to parametric memory (i.e., general hotel
knowledge baked into the model weights), forcing grounded answers.

### Method 2 — Confidence Gate

Before calling the LLM, the maximum retrieval similarity score is checked
against `SIMILARITY_THRESHOLD = 0.3`. If no chunk exceeds the threshold, the
system returns:

> "Insufficient context confidence. Cannot answer reliably."

This catches out-of-domain questions (e.g., "What is the stock price of
Marriott?") that would otherwise receive a hallucinated answer.

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
