"""
Hotel Q&A generator — strict RAG prompt + Groq LLM generation.

Strict prompting removes the model's fallback to parametric memory by
constraining answers to the provided context only.
"""
import logging
import os
import sys
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a hotel concierge assistant. Answer ONLY using the provided context. \
If the answer is not in the context, respond exactly with: \
"I don't have enough information to answer this from the available hotel data." \
For every fact you state, cite the source document ID in brackets like [hotel_003_chunk_2]."""


class HotelGenerator:
    """Generates answers to hotel queries using Groq LLM with strict RAG prompting."""

    def __init__(self):
        """Initialize Groq client from GROQ_API_KEY env variable."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        self.client = Groq(api_key=api_key)
        logger.info("Groq client initialized with model: %s", config.GROQ_MODEL)

    def build_prompt(self, query: str, chunks: list) -> str:
        """
        Build a strict RAG prompt from query and retrieved chunks.

        Args:
            query: User's hotel question.
            chunks: List of retrieved chunk dicts with chunk_id and text keys.

        Returns:
            Formatted prompt string with context blocks and question.

        Example:
            >>> gen = HotelGenerator()
            >>> prompt = gen.build_prompt("What is the WiFi policy?", chunks)
            >>> "CONTEXT:" in prompt and "QUESTION:" in prompt
            True
        """
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[{chunk['chunk_id']}] ({chunk['hotel_name']} — {chunk['category']})\n{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )
        return prompt

    def _extract_cited_sources(self, answer: str) -> list:
        """Parse bracketed chunk IDs from the answer text."""
        import re
        return re.findall(r'\[([^\]]+_chunk_\d+)\]', answer)

    def generate(self, query: str, chunks: list) -> dict:
        """
        Generate an answer for the query using retrieved chunks as context.

        Args:
            query: User's hotel question.
            chunks: List of retrieved chunk dicts (from HotelRetriever.retrieve).

        Returns:
            Dict with keys:
                answer (str): LLM-generated answer with inline citations.
                sources_cited (list[str]): Chunk IDs cited in the answer.
                prompt_used (str): Full user-turn prompt sent to the LLM.

        Example:
            >>> result = gen.generate("Do you have free WiFi?", chunks)
            >>> "answer" in result and "sources_cited" in result
            True
        """
        if not chunks:
            return {
                "answer": "I don't have enough information to answer this from the available hotel data.",
                "sources_cited": [],
                "prompt_used": "",
            }

        prompt = self.build_prompt(query, chunks)

        logger.info("Calling Groq API for query: '%s'", query[:60])
        response = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        sources_cited = self._extract_cited_sources(answer)

        logger.info("Generated answer (%d chars), cited %d sources", len(answer), len(sources_cited))
        return {
            "answer": answer,
            "sources_cited": sources_cited,
            "prompt_used": prompt,
        }

    def generate_unconstrained(self, query: str, chunks: list) -> dict:
        """
        Generate an answer WITHOUT strict context-only constraints (for comparison).

        Args:
            query: User's hotel question.
            chunks: List of retrieved chunk dicts.

        Returns:
            Same dict shape as generate(), but from an open-ended prompt.

        Example:
            >>> result = gen.generate_unconstrained("Do you have free WiFi?", chunks)
            >>> "answer" in result
            True
        """
        context_parts = [
            f"[{c['chunk_id']}] {c['text']}" for c in chunks
        ]
        context_block = "\n\n".join(context_parts)
        prompt = (
            f"Here is some hotel information:\n{context_block}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        response = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful hotel assistant. Answer the question."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip()
        return {
            "answer": answer,
            "sources_cited": self._extract_cited_sources(answer),
            "prompt_used": prompt,
        }
