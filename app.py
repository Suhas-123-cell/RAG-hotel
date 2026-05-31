"""
StayChat Hotel Q&A — Chat UI.

A conversational Gradio interface backed by the RAG pipeline.
Conversation history is kept in-session. Retrieval sources stay internal.
"""
import re
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
# Greeting / small-talk detection
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"h(i|ello|ey|owdy|iya)"
    r"|yo\b"
    r"|what'?s\s*up"
    r"|sup\b"
    r"|good\s*(morning|afternoon|evening|day)"
    r"|greetings"
    r"|who\s+are\s+you"
    r"|what\s+are\s+you"
    r"|what\s+can\s+you\s+do"
    r"|help"
    r"|thanks?"
    r"|thank\s*you"
    r"|bye|goodbye|see\s*ya"
    r"|ok(ay)?"
    r"|cool"
    r"|nice"
    r"|yes|no|yep|nope|yea|nah"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "👋 Hello! I'm **StayChat**, your hotel concierge assistant. "
    "I can answer questions about the hotels in our knowledge base, from luxury "
    "city stays to airport, capsule, heritage, mountain, and retreat properties.\n\n"
    "Try asking something like:\n"
    "- *Which hotels have free WiFi and complimentary breakfast?*\n"
    "- *What spa facilities does Serenity Palms offer?*\n"
    "- *Is The Azure Grand pet friendly?*"
)

_THANKS_RESPONSE = (
    "You're welcome! 😊 Feel free to ask anything else about our hotels."
)

_BYE_RESPONSE = (
    "Goodbye! 👋 Hope I was helpful. Come back anytime you need hotel info!"
)

def _detect_greeting(message: str) -> str | None:
    """Return a friendly response if the message is a greeting/small-talk, else None."""
    cleaned = message.strip().rstrip("!?. ")
    if not _GREETING_PATTERNS.match(message):
        return None
    lower = cleaned.lower()
    if lower in ("thanks", "thank you", "thank", "ty"):
        return _THANKS_RESPONSE
    if lower in ("bye", "goodbye", "see ya"):
        return _BYE_RESPONSE
    return _GREETING_RESPONSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_answer(answer: str) -> str:
    """Remove source markers from model output before showing it in chat."""
    answer = re.sub(r"\s*\[[^\]]*?_chunk_\d+\]", "", answer)
    answer = re.sub(r"\s*\[\d+\]", "", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def chat(message: str, history: list):
    message = message.strip()
    if not message:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please type a question about our hotels."},
        ]
        return history

    # Short-circuit for greetings / small-talk — skip the RAG pipeline entirely
    greeting_reply = _detect_greeting(message)
    if greeting_reply:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": greeting_reply},
        ]
        return history


    try:
        result = _pipeline.query(message)
    except Exception as exc:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Sorry, something went wrong: {exc}"},
        ]
        return history

    answer = _clean_answer(result["answer"])

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    return history


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

EXAMPLES = [
    "Which hotels have free WiFi and complimentary breakfast?",
    "Which hotels do not have televisions in the rooms?",
    "Which properties are best for late-night airport layovers?",
    "Compare bathroom essentials at The Azure Grand and Northstar Capsule Lodge.",
    "Which hotels are unsuitable for young children?",
    "Suggest a hotel for a noisy gaming weekend with very fast internet.",
]

CSS = """
    #chatbot { height: 520px; }
    footer { display: none !important; }
"""

with gr.Blocks(title="StayChat", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("# StayChat — Hotel Q&A")
    gr.Markdown(
        "Ask anything about the hotels in the knowledge base."
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        type="messages",
        allow_tags=False,
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

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot],
        outputs=chatbot,
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot],
        outputs=chatbot,
    ).then(fn=lambda: "", outputs=msg_box)

    clear_btn.click(fn=lambda: [], outputs=chatbot)

    gr.Markdown("---\n*Answers are grounded in the hotel knowledge base only.*")


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=config.GRADIO_PORT,
        server_name="127.0.0.1",
        show_error=True,
    )
