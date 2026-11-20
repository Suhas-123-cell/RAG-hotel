"""Report how much of each document's sentence text survives chunking.

Run from the repo root:
    python -m eval.chunk_coverage
"""
import json
import logging

import config
from src.embedder import HotelEmbedder
from src.preprocessor import HotelPreprocessor


def main() -> None:
    logging.disable(logging.CRITICAL)
    with open(config.DOCUMENTS_FILE, encoding="utf-8") as f:
        documents = json.load(f)
    pre = HotelPreprocessor()
    chunks = pre.process_all(documents, HotelEmbedder())

    by_doc: dict[str, list[str]] = {}
    for c in chunks:
        by_doc.setdefault(c["source_doc_id"], []).append(c["text"])

    docs_with_loss = 0
    total_tokens = 0
    lost_tokens = 0
    for doc in documents:
        cleaned = pre.clean_text(doc["text"])
        total_tokens += len(cleaned.split())
        covered = " ".join(by_doc.get(doc["id"], []))
        missing = [s for s in pre._split_into_sentences(cleaned) if s not in covered]
        if missing:
            docs_with_loss += 1
            lost_tokens += sum(len(s.split()) for s in missing)

    print(f"chunks: {len(chunks)}")
    print(f"documents with at least one sentence missing from their chunks: {docs_with_loss} of {len(documents)}")
    print(f"tokens missing from chunks: {lost_tokens} of {total_tokens} ({100 * lost_tokens / total_tokens:.1f}%)")


if __name__ == "__main__":
    main()
