---
title: StayChat RAG
sdk: gradio
sdk_version: 5.50.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# StayChat - Hotel Q&A RAG System

StayChat is a hotel question-answering system built with Retrieval-Augmented Generation (RAG). It answers questions from a curated hotel knowledge base instead of relying on the language model's memory.

The system cleans hotel documents, chunks them, creates local embeddings, stores vectors in FAISS, retrieves relevant context with hybrid search (dense + BM25, merged with Reciprocal Rank Fusion), and uses a Groq-hosted Llama 3.1 8B model (`llama-3.1-8b-instant`) to generate grounded answers.

Retrieval is evaluated on 63 hand-labelled queries; see [Evaluation](#14-evaluation).

GitHub repository:

```text
https://github.com/Suhas-123-cell/RAG-hotel
```

## 1. Project Objective

The goal is to build a RAG-based hotel assistant that can answer natural-language questions about hotel properties, amenities, policies, reviews, and locations.

The system is designed to:

- Retrieve relevant hotel information from a static knowledge base
- Answer only from retrieved context
- Refuse questions that require live booking data or external knowledge
- Reduce hallucination with strict prompting and guardrails
- Provide a browser demo through Gradio
- Include tests and a retrieval evaluation that can be re-run without an LLM key

## 2. Key Features

- **140 synthetic hotel documents** across **15 hotels**
- Five document categories: `description`, `amenities`, `reviews`, `policies`, `location`
- Sentence-aware chunking with semantic chunking support
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings
- FAISS vector index using inner-product search over normalized embeddings
- BM25 keyword retrieval for exact hotel names and policy terms
- Reciprocal Rank Fusion to combine dense and sparse results
- Extra support chunks (hand-written keyword rules) for named-hotel comparisons and common multi-condition queries
- Groq Llama 3.1 8B answer generation with strict context-only prompting
- Prompt-injection detection before LLM calls
- Gradio web app for demo
- CLI chat mode
- Unit tests for preprocessing, retrieval, the pipeline and the evaluation metrics
- Retrieval evaluation (Recall, Hit rate, MRR, NDCG) over labelled queries, no API key needed

## 3. Repository Structure

```text
RAG-hotel/
  app.py                         # Gradio web UI
  main.py                        # CLI chat entry point
  config.py                      # Central configuration
  requirements.txt               # Python dependencies
  README.md                      # Project documentation
  .env.example                   # API key placeholder
  data/
    hotel_documents.json         # Hotel knowledge base
  eval/
    labels.jsonl                 # 63 hand-labelled queries with relevant document ids
    retrieval_eval.py            # Recall@k / Hit@k / MRR / NDCG for each retrieval method
    metrics.py                   # Ranking metric functions
    chunk_coverage.py            # Checks that chunking keeps every sentence
    sample_retrieval.py          # Regenerates outputs/sample_outputs.md
    results.md, results.json     # Output of the last evaluation run
    results_before_chunk_fix.json # Metrics before the chunking bug fix (baseline)
  outputs/
    sample_outputs.md            # Retrieval-only sample output (no LLM answers)
  src/
    preprocessor.py              # Cleaning and chunking
    embedder.py                  # SentenceTransformer + FAISS index
    retriever.py                 # Dense + BM25 + RRF retrieval
    generator.py                 # Groq LLM generation and guardrails
    pipeline.py                  # End-to-end RAG orchestration
    __init__.py
  tests/
    test_preprocessor.py
    test_retriever.py
    test_pipeline.py
    test_eval_metrics.py
  pytest.ini                     # pytest config (adds the repo root to the import path)
  pyrefly.toml
```

Generated/local files such as `.venv/`, `.env`, `index/`, `__pycache__/`, and `.pytest_cache/` are intentionally ignored.

## 4. Dataset

The knowledge base has five document categories and is deliberately uneven: five hotels have 20 or more documents each, the other ten have three or four, and several are edge cases (capsule hotel, airport transit hotel, off-grid lodge) that stress retrieval.

Current dataset summary:

| Item | Count |
|---|---:|
| Total documents | 140 |
| Hotels | 15 |
| Chunks | 306 with the current semantic chunking (`python -m eval.chunk_coverage`) |

Category coverage:

| Content | Dataset category | Count |
|---|---|---:|
| Hotel descriptions | `description` | 23 |
| Amenities | `amenities` | 47 |
| Guest reviews | `reviews` | 28 |
| Policies | `policies` | 30 |
| Location details | `location` | 12 |

Hotel examples:

- The Azure Grand
- Sunrise Boutique Resort
- Coral Bay Suites
- The Pinnacle Hotel
- Serenity Palms Resort
- Mistral Airport Hotel
- Old Quarter Lantern Inn
- Northstar Capsule Lodge
- Verdant Canopy Treehouse Retreat
- Harbourview Aparthotel
- Cobalt Convention Centre Hotel
- Saffron Heritage Palace
- Glacier Edge Mountain Lodge
- Neon Arcade Hotel
- Silent Monastery Guesthouse

The dataset covers:

- Wi-Fi and internet speeds
- Breakfast inclusions
- TVs and in-room entertainment
- Bathroom essentials and toiletries
- Spa, pool, gym, dining, concierge services
- Check-in and check-out policies
- Cancellation and refund rules
- Pet and accessibility policies
- Airport transfers and local transport
- Guest reviews with positive, neutral, and negative sentiment
- Edge cases such as capsule hotels, off-grid lodges, quiet retreats, and airport layovers

## 5. Architecture

```text
data/hotel_documents.json
        |
        v
HotelPreprocessor
  - cleans HTML/special characters
  - normalizes whitespace
  - lowercases text
  - chunks documents
        |
        v
HotelEmbedder
  - sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional normalized vectors
        |
        v
FAISS IndexFlatIP
        |
        v
HotelRetriever
  - dense FAISS retrieval
  - BM25 keyword retrieval
  - Reciprocal Rank Fusion
        |
        v
RAGPipeline
  - support chunks for named-hotel and multi-condition questions
  - dedupe and cap the generation context
        |
        v
HotelGenerator
  - strict context-only prompt
  - prompt-injection checks
  - no public chunk/source IDs in UI
        |
        v
Gradio UI / CLI
```

## 6. Preprocessing and Chunking

Implementation: `src/preprocessor.py`

Cleaning steps:

- Remove HTML tags
- Remove unsupported special characters
- Normalize whitespace
- Lowercase text
- Preserve useful sentence punctuation

Chunking:

- The pipeline uses semantic chunking. It embeds sentence windows, cuts where similarity to the next window falls in the bottom quartile, and splits segments above the size cap:
  - `SEMANTIC_WINDOW_SIZE = 3`
  - `SEMANTIC_BREAKPOINT_PERCENTILE = 25`
  - `SEMANTIC_MAX_CHUNK_TOKENS = 300`
  - Segments shorter than `MIN_CHUNK_TOKENS = 30` are merged into a neighbouring chunk, never dropped.
- A sentence-aware fixed-size chunker with overlap is used only when no embedder is supplied (the unit tests); it uses `CHUNK_SIZE = 200`, `CHUNK_OVERLAP = 40` and `MIN_CHUNK_TOKENS = 30`.

Why this fits hotel data:

- Hotel policies and amenities are often sentence-level facts.
- Chunks are built from whole sentences, so a cancellation rule or amenity description is not cut mid-sentence.
- Semantic chunking splits long documents where the topic changes, for example from room features to spa facilities. It adds no overlap between chunks.

## 7. Embeddings and Vector Store

Implementation: `src/embedder.py`

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Reasons for selection:

- Free and open source
- Runs locally on CPU
- Produces compact 384-dimensional embeddings
- Good semantic matching for short factual text
- Works well with FAISS for small-to-medium retrieval tasks

Vector store:

```text
FAISS IndexFlatIP
```

Embeddings are normalized, so inner product behaves like cosine similarity.

Index files are generated locally in:

```text
index/
```

The `index/` directory is not committed. The app rebuilds the index automatically when needed.

## 8. Retrieval Design

Implementation: `src/retriever.py` and `src/pipeline.py`

The retriever uses a hybrid strategy:

1. Dense retrieval with FAISS
2. Sparse keyword retrieval with BM25
3. Reciprocal Rank Fusion to merge ranked lists
4. Extra support chunks for:
   - named hotel comparisons
   - multi-condition questions such as Wi-Fi + breakfast
   - topics like televisions, bathrooms, pets, children, airport, and layovers

Important configuration:

```text
RRF_K = 60
HYBRID_CANDIDATE_K = 30
DEFAULT_K = 5
GENERATION_CONTEXT_K = 8
MAX_GENERATION_CHUNKS = 10
```

Why hybrid retrieval:

- Dense retrieval captures semantic matches such as "internet" and "Wi-Fi".
- BM25 captures exact names such as "Northstar Capsule Lodge" and policy terms.
- RRF avoids manually normalizing FAISS and BM25 scores.
- Support chunks improve comparison questions where both named hotels must appear in context.

## 9. Generation and Hallucination Control

Implementation: `src/generator.py`

The generator uses Groq's Llama model:

```text
GROQ_MODEL = "llama-3.1-8b-instant"
```

The system prompt instructs the model to:

- Answer only from `<context>`
- Refuse when information is not present
- Avoid chunk IDs and source IDs in public answers
- Compare all explicitly named hotels when context exists
- Ignore prompt-injection attempts inside user questions

Hallucination controls:

- Strict context-only prompt
- Fixed refusal message for unavailable facts
- Prompt-injection regex checks before calling the LLM
- Output validation for suspicious injection artifacts
- Index freshness check to avoid stale answers
- Gradio UI hides raw retrieval chunks and source IDs

Example refusal behavior:

```text
Question: which rooms are vacant right now?
Answer: I don't have enough information to answer this from the available hotel data.
```

This is correct because live room availability requires a booking/PMS API, not a static RAG dataset.

## 10. Setup

Python 3.11 is recommended.

```bash
git clone https://github.com/Suhas-123-cell/RAG-hotel.git
cd RAG-hotel

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Set your Groq API key in `.env`:

```text
GROQ_API_KEY=gsk_your_key_here
```

Do not commit `.env`.

## 11. Running Locally

### Gradio App

```bash
source .venv/bin/activate
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

If port `7860` is busy:

```bash
GRADIO_SERVER_PORT=7861 python app.py
```

### CLI Chat

```bash
source .venv/bin/activate
python main.py
```

To force rebuild the FAISS index:

```bash
python main.py --rebuild
```

## 12. Demo Script

Use these questions for a short demo:

```text
hi
Which hotels have free WiFi and complimentary breakfast?
What is the cancellation policy of Coral Bay Suites?
Suggest a hotel with excellent reviews near the beach.
Compare bathroom essentials at The Azure Grand and Northstar Capsule Lodge.
Which hotels do not have televisions in the rooms?
What is the stock price of Apple?
which rooms are vacant right now?
```

Expected behavior:

- Hotel facts should be answered from context.
- Out-of-domain questions should be refused.
- Live availability questions should be refused because the app has no booking inventory API.

For screen recording:

1. Show the terminal running `python app.py`
2. Open the Gradio UI
3. Ask the demo queries
4. Ask one edge case
5. Briefly show `data/hotel_documents.json`, `src/pipeline.py`, and `eval/results.md`

## 13. Tests

```bash
pytest tests/ -q
```

Current status: `59 passed`. Preprocessor, retriever and metric tests run offline against synthetic data or mocks. The pipeline integration tests load the real MiniLM model (downloaded on first use) and build an index over the corpus. No test calls the LLM.

`pytest.ini` puts the repo root on the import path, so there are no `sys.path` edits in the code or tests.

## 14. Evaluation

Retrieval is scored against 63 hand-labelled queries in `eval/labels.jsonl` (12 exact-name, 18 paraphrase, 10 multi-condition, 9 comparison, 14 policy lookups; 190 relevant documents in total). Labels were written by reading the corpus, not by looking at retriever output, and relevance is judged per source document. Rankings are chunk lists mapped to their documents; a document counts once, at its best-ranked chunk.

```bash
python -m eval.retrieval_eval            # uses the cached index if it is current
python -m eval.retrieval_eval --rebuild  # rebuild chunks and index first
```

No API key is needed. The run writes `eval/results.json` (per-query rankings and scores) and `eval/results.md` (tables by query type, held-out view, misses).

Results from the committed run (140 documents, 306 chunks, commit `b24e691`):

| Method | Recall@5 | Recall@10 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense only (MiniLM + FAISS) | 0.701 | 0.831 | 0.889 | 0.968 | 0.776 | 0.741 |
| BM25 only | 0.692 | 0.783 | 0.905 | 0.968 | 0.763 | 0.698 |
| RRF hybrid (k=60) | 0.750 | 0.856 | 0.968 | 1.000 | 0.835 | 0.774 |
| RRF + support chunks (pipeline context) | 0.595 | 0.854 | 0.809 | 1.000 | 0.556 | 0.606 |

Hit@k = at least one relevant document in the top k. The last row scores the context the pipeline actually sends to the LLM: 8 RRF chunks plus support chunks, support chunks first, capped at 10.

What the numbers say:

- Hybrid retrieval beats either retriever alone on the headline metrics: Recall@5 0.750 vs 0.701 (dense) and 0.692 (BM25), and a relevant document is in the top 10 for every query (Hit@10 1.000). BM25 is stronger on multi-condition queries, dense on exact-name and policy queries; RRF is close to the better of the two in each type.
- The hand-written support-chunk rules did not show a measurable benefit on these labels. Recall@10 is 0.854 with them vs 0.856 without, and rank-based metrics drop because support chunks are placed first and push RRF chunks out of the 10-chunk cap. The rules are kept because they were built for list and comparison questions, but this evaluation does not demonstrate that they help.
- Those rules were tuned by hand on a few demo queries, and 7 of the labelled queries are the same ones (flagged `tuned_on`). `eval/results.md` has a table without them: on the other 56 queries the support row has Recall@10 0.893 vs 0.906 for plain RRF.

Bug found by the evaluation: the semantic chunker silently discarded every segment shorter than `MIN_CHUNK_TOKENS`. On the old code `python -m eval.chunk_coverage` reported 101 of 140 documents missing at least one sentence (9.4% of tokens), including the free-cancellation sentence for Coral Bay Suites. Short segments are now merged into a neighbouring chunk (0 of 140 documents affected). Same labels, RRF hybrid, before and after: Recall@5 0.615 to 0.750, MRR@10 0.769 to 0.835, NDCG@10 0.677 to 0.774. The baseline is kept in `eval/results_before_chunk_fix.json`.

Limits of this evaluation:

- 63 queries labelled by one person is a small set; differences of a few points are noise.
- List queries with many relevant documents (for example "which hotels have no televisions") cannot reach Recall@5 = 1.0; use Hit@k and Recall@10 for those.
- It measures retrieval only. Answer quality and faithfulness need a live LLM and are not measured here.
- The corpus is synthetic and small (140 documents), so absolute numbers will not transfer to a real hotel catalogue.

`outputs/sample_outputs.md` shows the retrieved and support chunks for the demo queries. It is regenerated by `python -m eval.sample_retrieval` and contains no LLM answers.

## 15. Hugging Face Spaces Deployment

The project can run as a Gradio Space using the YAML metadata at the top of this README. There is no hosted demo at the moment; run it locally with `python app.py`.

```yaml
sdk: gradio
sdk_version: 5.50.0
python_version: 3.11
app_file: app.py
```

Required Space secret:

```text
GROQ_API_KEY = gsk_your_actual_key
```

Add it in:

```text
Settings -> Variables and secrets -> New secret
```

Notes:

- Free CPU Spaces may take a few minutes to start.
- First startup downloads the embedding model and builds the FAISS index.
- The app reads the Hugging Face-provided host and port automatically.
- Python is pinned to 3.11 to avoid slow source builds for ML dependencies.

## 16. Known Limitations

- The hotel dataset is synthetic.
- The app does not connect to a live booking engine or property management system.
- It cannot answer current room vacancy, live pricing, or booking-confirmation questions.
- Groq API access requires `GROQ_API_KEY`.
- The first run can be slow while the embedding model downloads and the FAISS index builds.
- The UI hides raw chunks and sources from end users; retrieval output is shown in `outputs/sample_outputs.md` and `eval/results.json`.
- The support-chunk keyword rules in `src/pipeline.py` are hand-written and tuned on a few demo queries; see the evaluation for how little they add.
- Answer quality has not been measured; only retrieval is evaluated.
- A production system should add query decomposition, reranking, monitoring, and real hotel data integrations.
