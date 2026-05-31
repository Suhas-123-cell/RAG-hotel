"""
StayChat Hotel Q&A — Chat UI.

A conversational Gradio interface backed by the RAG pipeline.
Conversation history is kept in-session. Retrieved source chunks
are shown in a collapsible panel alongside the chat.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import config
from src.pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline singleton — built once, reused for every message
# ---------------------------------------------------------------------------

print("Loading StayChat… (first run builds the search index, ~30s)")
_pipeline = RAGPipeline(force_rebuild=False)
print(f"Ready — {len(_pipeline.chunks)} chunks indexed.\n")

# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def chat(message: str, history: list):
    """
    Handle one chat turn.

    Args:
        message: Latest user message string.
        history: List of [user, assistant] string pairs (Gradio Chatbot state).

    Returns:
        Tuple of (updated_history, sources_markdown_string).
    """
    message = message.strip()
    if not message:
        history = history + [[message, "Please type a question about our hotels."]]
        return history, ""

    try:
        result = _pipeline.query(message)
    except Exception as exc:
        history = history + [[message, f"Sorry, something went wrong: {exc}"]]
        return history, ""

    answer = result["answer"]

    sources = result.get("sources_cited", [])
    if sources:
        answer += "\n\n**Sources:** " + " · ".join(f"`{s}`" for s in sources)

    history = history + [[message, answer]]

    chunks = result.get("retrieved_chunks", [])
    if chunks:
        lines = ["### Retrieved chunks\n"]
        for i, c in enumerate(chunks, 1):
            score = c.get("rrf_score") or c.get("score", 0.0)
            text_preview = c["text"][:200].replace("\n", " ")
            lines.append(
                f"**[{i}] {c['chunk_id']}**  \n"
                f"`{c['hotel_name']}` · `{c['category']}` · score `{score:.4f}`  \n"
                f"{text_preview}…\n"
            )
        sources_md = "\n---\n".join(lines)
    else:
        sources_md = "No chunks retrieved."

    return history, sources_md


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

EXAMPLES = [
    "Which hotels have free WiFi and complimentary breakfast?",
    "What is the cancellation policy of Coral Bay Suites?",
    "Suggest a hotel with excellent reviews near the beach.",
    "What spa and wellness facilities does Serenity Palms offer?",
    "Is The Azure Grand pet friendly?",
    "How far is Sunrise Boutique Resort from the airport?",
]

with gr.Blocks(title="StayChat", theme=gr.themes.Soft(), css="""
    #chatbot { height: 500px; }
    #sources-panel { height: 500px; overflow-y: auto; }
    footer { display: none !important; }
""") as demo:

    gr.Markdown("# StayChat — Hotel Q&A")
    gr.Markdown(
        "Ask anything about **The Azure Grand**, **Sunrise Boutique Resort**, "
        "**Coral Bay Suites**, **The Pinnacle Hotel**, or **Serenity Palms Resort**."
    )

    with gr.Row():

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=False,
                show_label=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a hotel question and press Enter…",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[[e] for e in EXAMPLES],
                inputs=msg_box,
                label="Example questions",
            )

            clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

        with gr.Column(scale=2):
            with gr.Accordion("Retrieved sources (last query)", open=True):
                sources_box = gr.Markdown(
                    value="*Ask a question to see which hotel documents were retrieved.*",
                    elem_id="sources-panel",
                )

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, sources_box],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, sources_box],
    ).then(fn=lambda: "", outputs=msg_box)

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, sources_box])

    gr.Markdown("---\n*Answers are grounded in the hotel knowledge base only.*")


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=config.GRADIO_PORT,
        server_name="0.0.0.0",
        show_error=True,
    )
