# StayChat - Hotel Q&A RAG System

StayChat is a Retrieval-Augmented Generation (RAG) system for answering natural-language questions about hotels. It cleans and chunks a hotel knowledge base, embeds the chunks locally, stores them in FAISS, retrieves relevant context with hybrid search, and asks an LLM to answer only from the retrieved hotel data.

This project was built for the StayChat AI / ML developer assessment.

## What This Project Includes

- A synthetic hotel knowledge base in `data/hotel_documents.json`
- Text cleaning and chunking in `src/preprocessor.py`
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector search in `src/embedder.py`
- Hybrid retrieval using dense FAISS search plus BM25 in `src/retriever.py`
- Context-grounded answer generation with Groq Llama in `src/generator.py`
- End-to-end orchestration in `src/pipeline.py`
- Gradio demo app in `app.py`
- CLI chat entry point in `main.py`
- Unit tests in `tests/`
- Sample assessment outputs in `outputs/sample_outputs.md`

## Current Dataset

The dataset currently contains **140 hotel documents** across **15 hotels**. It is intentionally uneven to stress-test retrieval with a more realistic corpus: some hotels have many detailed records, while others have sparse or niche records.

Hotel examples include:

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

Document categories:

- `description`
- `amenities`
- `reviews`
- `policies`
- `location`

The data covers Wi-Fi, breakfast, room amenities, TVs, bathroom essentials, check-in/check-out rules, cancellation policies, pet policies, accessibility, airport transfers, family suitability, wellness retreats, capsule/shared bathrooms, off-grid stays, convention hotels, and edge cases.

## Architecture

```text
hotel_documents.json
        |
        v
HotelPreprocessor
  - clean text
  - sentence-aware chunking
  - semantic chunking when embedder is available
        |
        v
HotelEmbedder
  - sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
        |
        v
FAISS IndexFlatIP
        |
        v
HotelRetriever
  - dense FAISS retrieval
  - BM25 sparse retrieval
  - Reciprocal Rank Fusion
  - extra support chunks for named-hotel and multi-condition queries
        |
        v
HotelGenerator
  - strict context-only prompt
  - prompt-injection checks
  - no public source/chunk IDs in the UI answer
        |
        v
Gradio UI / CLI
```

## Tech Stack

| Component | Library / Service | Purpose |
|---|---|---|
| Python | 3.11 recommended | Runtime |
| Embeddings | `sentence-transformers` | Local semantic embeddings |
| Vector store | `faiss-cpu` | Similarity search |
| Sparse retrieval | `rank-bm25` | Keyword retrieval |
| LLM | Groq API | Answer generation |
| UI | Gradio | Browser demo |
| Tests | pytest | Unit tests |
| Env loading | python-dotenv | API key loading |

## Setup

Use Python 3.11 if possible. Some ML packages may not support the newest Python versions immediately.

```bash
git clone https://github.com/<your-username>/RAG-hotel.git
cd RAG-hotel

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GROQ_API_KEY=gsk_your_key_here
```

## Run the Gradio Demo

```bash
source .venv/bin/activate
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

On first launch, the system builds the FAISS index. If `data/hotel_documents.json` is newer than the cached index, the app rebuilds automatically.

To force a rebuild from the CLI:

```bash
python main.py --rebuild
```

## Demo Script

Use the Gradio UI for the live demo. A good 3-5 minute demo flow:

1. Start the app:

   ```bash
   python app.py
   ```

2. Open `http://127.0.0.1:7860`.

3. Ask a greeting:

   ```text
   hi
   ```

   This shows the assistant is conversational and does not call retrieval for simple greetings.

4. Ask the required assessment query:

   ```text
   Which hotels have free WiFi and complimentary breakfast?
   ```

5. Ask a comparison query:

   ```text
   Compare bathroom essentials at The Azure Grand and Northstar Capsule Lodge.
   ```

6. Ask an edge-case query:

   ```text
   Which hotels do not have televisions in the rooms?
   ```

7. Ask an out-of-domain query:

   ```text
   What is the stock price of Apple?
   ```

   The expected behavior is a refusal or "not enough information" answer because the knowledge base is only about hotels.

Recommended screen recording:

- Show terminal running `python app.py`
- Show browser at `http://127.0.0.1:7860`
- Ask the four demo questions above
- Briefly show `data/hotel_documents.json`, `src/pipeline.py`, and `outputs/sample_outputs.md`

## Run CLI Chat

```bash
source .venv/bin/activate
python main.py
```

Then type questions interactively. Use `quit`, `exit`, or `q` to stop.

## Run Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Fast focused test run:

```bash
pytest tests/test_preprocessor.py tests/test_retriever.py -q
```

## Retrieval Design

The retriever combines:

- Dense semantic search over FAISS
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)

This helps with both semantic questions and exact-match hotel details. For example, dense retrieval helps with paraphrases like "internet access" versus "Wi-Fi", while BM25 helps with exact hotel names, policies, and terms like "Northstar Capsule Lodge".

Important config values are in `config.py`:

- `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`
- `EMBEDDING_DIM = 384`
- `HYBRID_CANDIDATE_K = 50`
- `GENERATION_CONTEXT_K = 14`
- `RRF_K = 60`
- `DEFAULT_K = 5`

## Hallucination Control

The system uses several controls:

- Strict context-only system prompt
- Fixed refusal when the answer is not in retrieved context
- Prompt-injection pattern checks before calling the LLM
- Output validation for suspicious prompt-injection artifacts
- Hidden retrieval context instead of public chunk IDs in the Gradio UI
- Automatic index freshness check so the app does not accidentally answer from stale data

The Gradio UI intentionally does not show raw chunk IDs or retrieved source chunks to end users. The sample-output file still documents retrieval behavior for assessment review.

## Sample Outputs and Evaluation

Assessment sample outputs are in:

```text
outputs/sample_outputs.md
```

That file should include:

- The three required example queries
- Retrieved chunks or summaries
- Final LLM answers
- Retrieval metric results such as Precision@k or MRR
- A short qualitative analysis and at least one edge case

## GitHub Submission

Before pushing, make sure these files are included:

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
```

Do not commit these:

```text
.env
.venv/
__pycache__/
.pytest_cache/
index/
```

Push to GitHub:

```bash
git init
git add .
git commit -m "Add StayChat hotel RAG system"
git branch -M main
git remote add origin https://github.com/<your-username>/RAG-hotel.git
git push -u origin main
```

If the repo already exists locally and has a remote:

```bash
git add .
git commit -m "Finalize StayChat RAG submission"
git push
```

Submission options from the assessment:

- Submit the GitHub repository link, or
- Submit a single ZIP archive containing the project

To create a ZIP:

```bash
cd ..
zip -r RAG-hotel-submission.zip RAG-hotel \
  -x "RAG-hotel/.venv/*" \
  -x "RAG-hotel/.env" \
  -x "RAG-hotel/index/*" \
  -x "RAG-hotel/__pycache__/*" \
  -x "RAG-hotel/.pytest_cache/*"
```

## Deploy on Hugging Face Spaces

You can deploy the Gradio app on a free Hugging Face account using **Spaces**.

### Option A - Deploy from the Hugging Face Website

1. Create or log in to your Hugging Face account.

2. Go to:

   ```text
   https://huggingface.co/new-space
   ```

3. Create a new Space:

   ```text
   Space name: staychat-rag
   License: MIT or other
   SDK: Gradio
   Hardware: CPU basic - free
   Visibility: Public or Private
   ```

4. Upload or push these project files to the Space:

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
   ```

5. Do not upload:

   ```text
   .env
   .venv/
   index/
   __pycache__/
   .pytest_cache/
   ```

6. Add your Groq API key as a Space secret:

   - Open your Space on Hugging Face
   - Go to **Settings**
   - Go to **Variables and secrets**
   - Add a new **Secret**

   ```text
   Name: GROQ_API_KEY
   Value: gsk_your_actual_key_here
   ```

   Do not put the real key in `app.py`, `README.md`, or `.env`.

7. Wait for the Space to build.

   The first build can take several minutes because it installs dependencies and downloads the embedding model. The first app startup can also take time because the FAISS index is built from `data/hotel_documents.json`.

8. Once it says **Running**, open the public Space URL:

   ```text
   https://huggingface.co/spaces/<your-username>/staychat-rag
   ```

### Option B - Deploy from Terminal with Git

Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

Log in:

```bash
huggingface-cli login
```

Create a Space from the website first, then clone it:

```bash
git clone https://huggingface.co/spaces/<your-username>/staychat-rag
cd staychat-rag
```

Copy this project into the Space folder, excluding local generated files:

```bash
rsync -av --exclude ".git" --exclude ".venv" --exclude ".env" --exclude "index" \
  --exclude "__pycache__" --exclude ".pytest_cache" \
  /path/to/RAG-hotel/ ./
```

Commit and push:

```bash
git add .
git commit -m "Deploy StayChat RAG Gradio app"
git push
```

Then add the `GROQ_API_KEY` secret in the Space settings.

### Option C - Deploy with `gradio deploy`

Gradio also supports direct deployment:

```bash
source .venv/bin/activate
gradio deploy
```

Follow the prompts. The deploy command uploads the app to Hugging Face Spaces and respects `.gitignore`.

### Hugging Face Space Notes

- The Space must have `requirements.txt` at the repo root.
- The main file should be named `app.py`.
- This project reads `GROQ_API_KEY` from environment variables, which is how Hugging Face exposes Space secrets.
- `app.py` is configured to use the server host/port provided by Spaces.
- The free CPU Space is enough for a small demo, but startup may be slow because `sentence-transformers` and FAISS run on CPU.
- If the Space fails with an API key error, check that `GROQ_API_KEY` is added as a **Secret**, not as plain text in the code.
- If the Space rebuilds repeatedly, check the build logs from the Space page.

Recommended Space demo questions:

```text
Which hotels have free WiFi and complimentary breakfast?
Compare bathroom essentials at The Azure Grand and Northstar Capsule Lodge.
Which hotels do not have televisions in the rooms?
Which properties are best for late-night airport layovers?
What is the stock price of Apple?
```

## Known Limitations

- The dataset is synthetic, not scraped from real hotel websites.
- Groq requires an API key in `.env`.
- The first run may take time because embeddings and the FAISS index are built locally.
- If the embedding model is not cached, the machine needs internet access on first run to download it.
- The UI hides source chunks for user experience, but sample outputs still document retrieval details for assessment.
- Multi-hop retrieval is improved with support chunks, but a production system should add query decomposition or a cross-encoder re-ranker.

## Assessment Checklist

- [x] Source code included
- [x] Dataset included
- [x] README included
- [x] `requirements.txt` included
- [x] Gradio demo included
- [x] Preprocessing implemented
- [x] Vector embeddings and FAISS implemented
- [x] Hybrid retrieval implemented
- [x] LLM generation implemented
- [x] Hallucination controls implemented
- [x] Tests included
- [x] Sample outputs included
