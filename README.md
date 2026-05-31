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

The system cleans hotel documents, chunks them, creates local embeddings, stores vectors in FAISS, retrieves relevant context with hybrid search, and uses a Groq-hosted Llama model to generate grounded answers.

Live demo:

```text
https://huggingface.co/spaces/suhas20sh/staychat-rag
```

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
- Include tests and reproducible setup instructions

## 2. Key Features

- **140 synthetic hotel documents** across **15 hotels**
- Five required document categories: `description`, `amenities`, `reviews`, `policies`, `location`
- Sentence-aware chunking with semantic chunking support
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings
- FAISS vector index using inner-product search over normalized embeddings
- BM25 keyword retrieval for exact hotel names and policy terms
- Reciprocal Rank Fusion to combine dense and sparse results
- Extra support chunks for named-hotel comparisons and common multi-condition queries
- Groq Llama answer generation with strict context-only prompting
- Prompt-injection detection before LLM calls
- Gradio web app for demo
- CLI chat mode
- Unit tests for preprocessing and retrieval

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
  outputs/
    sample_outputs.md            # Sample answers and evaluation notes
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
  pytest.ini
  pyrefly.toml
```

Generated/local files such as `.venv/`, `.env`, `index/`, `__pycache__/`, and `.pytest_cache/` are intentionally ignored.

## 4. Dataset

The assessment asks for 30-50 hotel documents across five categories. This project intentionally expands beyond the minimum to stress-test retrieval on a harder, uneven dataset.

Current dataset summary:

| Item | Count |
|---|---:|
| Total documents | 140 |
| Hotels | 15 |
| Chunked records | About 296 on current semantic chunking |

Category coverage:

| Required category | Dataset category | Current count |
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
  - support chunks for named comparisons
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

- Fallback chunking is sentence-aware and uses:
  - `CHUNK_SIZE = 200`
  - `CHUNK_OVERLAP = 40`
  - `MIN_CHUNK_TOKENS = 30`
- Semantic chunking uses sentence-window embeddings and breakpoints:
  - `SEMANTIC_WINDOW_SIZE = 3`
  - `SEMANTIC_BREAKPOINT_PERCENTILE = 25`
  - `SEMANTIC_MAX_CHUNK_TOKENS = 300`

Why this fits hotel data:

- Hotel policies and amenities are often sentence-level facts.
- Sentence-aware chunking avoids cutting cancellation rules or amenity descriptions mid-thought.
- Overlap preserves context across boundaries.
- Semantic chunking helps split long documents when the topic changes from, for example, room features to spa facilities.

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
3. Ask the assessment queries
4. Ask one edge case
5. Briefly show `data/hotel_documents.json`, `src/pipeline.py`, and `outputs/sample_outputs.md`

## 13. Tests

Run all tests:

```bash
pytest tests/ -v
```

Fast focused tests:

```bash
pytest tests/test_preprocessor.py tests/test_retriever.py -q
```

Current focused test status:

```text
30 passed
```

## 14. Evaluation and Sample Outputs

Sample assessment outputs are in:

```text
outputs/sample_outputs.md
```

This file contains:

- Required example queries
- Retrieved chunks or summaries
- Final answers
- Retrieval metric results
- Qualitative analysis
- Failure or edge-case notes

The assessment asked for at least one retrieval metric such as Precision@k, Recall@k, or MRR. This project reports Precision@k and MRR-style analysis in the sample outputs.

## 15. Hugging Face Spaces Deployment

The project is configured for Hugging Face Spaces with the YAML metadata at the top of this README:

```yaml
sdk: gradio
sdk_version: 5.50.0
python_version: 3.11
app_file: app.py
```

The live Space is:

```text
https://huggingface.co/spaces/suhas20sh/staychat-rag
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

## 16. GitHub Submission

Files that should be included:

```text
app.py
main.py
config.py
requirements.txt
README.md
.env.example
data/hotel_documents.json
outputs/sample_outputs.md
src/
tests/
pytest.ini
pyrefly.toml
```

Files that should not be committed:

```text
.env
.venv/
index/
__pycache__/
.pytest_cache/
```

Submit either:

- GitHub repository link
- ZIP archive of the project

Suggested email:

```text
Subject: StayChat AI Developer Assessment Submission - Suhas

Hi [Name],

Please find my StayChat RAG assessment submission below:

GitHub Repository:
https://github.com/Suhas-123-cell/RAG-hotel

Live Gradio Demo:
https://huggingface.co/spaces/suhas20sh/staychat-rag

The project includes source code, dataset, README, requirements.txt, tests, sample outputs, and a live Gradio demo. The Hugging Face Space may take a minute to wake up on the free CPU tier.

Thank you,
Suhas
```

## 17. Known Limitations

- The hotel dataset is synthetic.
- The app does not connect to a live booking engine or property management system.
- It cannot answer current room vacancy, live pricing, or booking-confirmation questions.
- Groq API access requires `GROQ_API_KEY`.
- The first run can be slow while the embedding model downloads and the FAISS index builds.
- Free Hugging Face Spaces can sleep and may need time to restart.
- The UI hides raw chunks for end users, although sample outputs document retrieval behavior for assessment.
- A production system should add query decomposition, reranking, monitoring, and real hotel data integrations.

## 18. Assessment Checklist

- [x] Source code
- [x] Dataset
- [x] README
- [x] `requirements.txt`
- [x] Gradio demo
- [x] Preprocessing
- [x] Embeddings
- [x] FAISS vector store
- [x] Hybrid retrieval
- [x] LLM generation
- [x] Hallucination controls
- [x] Tests
- [x] Sample outputs
