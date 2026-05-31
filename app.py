"""
StayChat Hotel Q&A — Gradio web interface.

Initialises the RAG pipeline once at module level and exposes a
Gradio Blocks UI with three output panels:
  1. Answer (with inline citations)
  2. Retrieved Chunks (formatted text)
  3. Confidence Score (top chunk similarity)
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

import config
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline — initialised once, reused across all UI requests
# ---------------------------------------------------------------------------
logger.info("Initialising RAGPipeline for Gradio app…")
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Lazy singleton so the pipeline is only built once even in reload scenarios."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        _pipeline = RAGPipeline(force_rebuild=False)
        logger.info("Pipeline ready — %d chunks indexed.", len(_pipeline.chunks))
    return _pipeline


# ---------------------------------------------------------------------------
# Interface function
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list) -> str:
    """Format retrieved chunks into a readable multi-line string."""
    if not chunks:
        return "No chunks retrieved."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0.0)
        hotel = chunk.get("hotel_name", "Unknown")
        category = chunk.get("category", "—")
        chunk_id = chunk.get("chunk_id", "—")
        text = chunk.get("text", "").strip()

        lines.append(
            f"[{i}] {chunk_id}\n"
            f"    Hotel    : {hotel}\n"
            f"    Category : {category}\n"
            f"    Score    : {score:.4f}\n"
            f"    Text     : {text[:300]}{'…' if len(text) > 300 else ''}"
        )
        lines.append("")  # blank line between chunks

    return "\n".join(lines).strip()


def answer_question(question: str):
    """
    Main handler called by the Gradio UI.

    Args:
        question: The user's hotel question.

    Returns:
        Tuple of (answer_with_sources, formatted_chunks, confidence_score_str).
    """
    question = question.strip()
    if not question:
        return (
            "Please enter a question about our hotels.",
            "No question provided.",
            "N/A",
        )

    try:
        pipeline = _get_pipeline()
        result = pipeline.query(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline query failed")
        return (
            f"An error occurred while processing your question:\n{exc}",
            "Retrieval unavailable.",
            "N/A",
        )

    # --- Answer panel ---
    answer = result["answer"]
    sources = result.get("sources_cited", [])
    if sources:
        answer += f"\n\nSources cited: {', '.join(sources)}"

    # --- Chunks panel ---
    chunks_text = _format_chunks(result.get("retrieved_chunks", []))

    # --- Confidence panel ---
    top_score = result.get("top_score", 0.0)
    threshold = config.SIMILARITY_THRESHOLD
    passed = len(result.get("filtered_chunks", []))
    total = len(result.get("retrieved_chunks", []))
    confidence_str = (
        f"{top_score:.4f}  "
        f"(threshold={threshold}, {passed}/{total} chunks passed filter)"
    )

    return answer, chunks_text, confidence_str


# ---------------------------------------------------------------------------
# Gradio Blocks layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title=config.GRADIO_TITLE,
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(f"# {config.GRADIO_TITLE}")
    gr.Markdown(
        "Ask any question about our hotels — policies, amenities, reviews, "
        "location, and more. The system retrieves relevant passages from our "
        "hotel knowledge base and generates a grounded answer."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Ask a hotel question",
                placeholder=(
                    "e.g. Which hotels have free WiFi and complimentary breakfast?"
                ),
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("**Example questions**")
            gr.Examples(
                examples=[
                    ["Which hotels have free WiFi and complimentary breakfast?"],
                    ["What is the cancellation policy of Coral Bay Suites?"],
                    ["Suggest a hotel with excellent reviews near the beach."],
                    ["What wellness amenities does Serenity Palms offer?"],
                    ["Does The Azure Grand have a restaurant?"],
                ],
                inputs=question_input,
            )

    with gr.Row():
        answer_output = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False,
        )

    with gr.Row():
        with gr.Column():
            chunks_output = gr.Textbox(
                label="Retrieved Chunks",
                lines=14,
                interactive=False,
            )
        with gr.Column(scale=1):
            confidence_output = gr.Textbox(
                label="Confidence Score",
                lines=2,
                interactive=False,
            )

    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, chunks_output, confidence_output],
    )
    question_input.submit(
        fn=answer_question,
        inputs=question_input,
        outputs=[answer_output, chunks_output, confidence_output],
    )

    gr.Markdown(
        "---\n*StayChat is a demonstration RAG system. "
        "Answers are grounded in retrieved hotel documents only.*"
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pre-warm the pipeline so the first request isn't slow
    _get_pipeline()
    demo.launch(
        share=False,
        server_port=config.GRADIO_PORT,
        server_name="0.0.0.0",
    )
