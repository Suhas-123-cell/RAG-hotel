"""StayChat — Hotel Q&A RAG System CLI entry point."""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BANNER = r"""
 ____  _              ____ _           _
/ ___|| |_ __ _ _   _/ ___| |__   __ _| |_
\___ \| __/ _` | | | \___ \ '_ \ / _` | __|
 ___) | || (_| | |_| |___) | | | | (_| | |_
|____/ \__\__,_|\__, |____/|_| |_|\__,_|\__|
                |___/
  Hotel Q&A — Powered by RAG + Groq LLaMA-3
"""

SEPARATOR = "=" * 70
THIN_SEP = "-" * 70


def _check_groq_key() -> None:
    """Raise a clear error if GROQ_API_KEY is missing from the environment."""
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("\n[ERROR] GROQ_API_KEY is not set.")
        print("  1. Copy .env.example to .env  (cp .env.example .env)")
        print("  2. Add your Groq API key:     GROQ_API_KEY=gsk_...")
        print("  3. Re-run this script.\n")
        sys.exit(1)


def run_chat_mode(pipeline) -> None:
    """Interactive question-answer loop using the loaded pipeline."""
    print(f"\n{SEPARATOR}")
    print("  INTERACTIVE CHAT MODE")
    print("  Type your hotel question and press Enter.")
    print("  Commands: 'quit' / 'exit' / 'q' to stop, 'help' for tips.")
    print(SEPARATOR)

    tips = (
        "  Try asking:\n"
        "    - Which hotels have free WiFi and complimentary breakfast?\n"
        "    - What is the cancellation policy of Coral Bay Suites?\n"
        "    - Suggest a hotel with excellent reviews near the beach.\n"
        "    - What wellness amenities does Serenity Palms offer?\n"
    )

    while True:
        try:
            print()
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not question:
            continue

        lower_q = question.lower()
        if lower_q in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            break
        if lower_q == "help":
            print(tips)
            continue

        print()
        try:
            result = pipeline.query(question)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] Query failed: {exc}")
            continue

        print(f"  {THIN_SEP}")
        print(f"  ANSWER:")
        for line in result["answer"].splitlines():
            print(f"    {line}")

        if result["sources_cited"]:
            print(f"\n  SOURCES: {', '.join(result['sources_cited'])}")

        print(f"  CONFIDENCE (top score): {result['top_score']:.4f}")
        print(f"  Retrieved {len(result['retrieved_chunks'])} chunks, "
              f"{len(result['filtered_chunks'])} passed threshold.")
        print(f"  {THIN_SEP}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="StayChat Hotel Q&A — RAG Pipeline CLI",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of the FAISS index from source documents",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    print(BANNER)
    _check_groq_key()

    print(f"  Rebuild: {args.rebuild}")
    print(f"  Log    : {args.log_level}")
    print(f"\n{THIN_SEP}")

    try:
        from src.pipeline import RAGPipeline
    except ImportError as exc:
        print(f"\n[ERROR] Failed to import RAGPipeline: {exc}")
        print("  Run: pip install -r requirements.txt\n")
        sys.exit(1)

    try:
        print("\n  Initializing pipeline...")
        pipeline = RAGPipeline(force_rebuild=args.rebuild)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Pipeline initialization failed: {exc}\n")
        sys.exit(1)

    print(f"\n{SEPARATOR}")
    print(f"  Pipeline ready — {len(pipeline.chunks)} chunks indexed.")
    print(SEPARATOR)

    try:
        run_chat_mode(pipeline)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Goodbye!\n")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Unexpected failure: {exc}\n")
        logging.exception("Unhandled exception")
        sys.exit(1)

    print(f"\n{SEPARATOR}")
    print("  StayChat session complete.")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
