"""
state.py
--------
Defines the AgentState that flows through every node in the LangGraph.
Every node reads from this state and writes back to it.
"""

from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────
    query: str                          # original user question

    # # ── Query analysis ────────────────────────────────────────
    # language: str                       # "en" | "hi" | "mixed"
    # intent: str                         # "factual" | "eligibility" | "procedural" | "comparison"
    
    # ----- Query refinement ------------------------------------
    rewritten_query: str                # query after rewriting (if triggered)
    rewrite_count: int                  # how many times rewritten (max 2)

    # ── Retrieval ─────────────────────────────────────────────
    retrieved_docs: list[dict]          # raw retrieved chunks from Qdrant
    relevant_docs: list[dict]           # after relevance grading

    # ── Generation ────────────────────────────────────────────
    answer: str                         # final generated answer
    sources: list[dict]                 # source citations

    # ── Grading ───────────────────────────────────────────────
    relevance_grade: str                # "sufficient" | "insufficient"
    hallucination_grade: str            # "grounded" | "hallucinated"
    answer_grade: str                   # "answers_question" | "does_not_answer"

    # ── Conversation memory ───────────────────────────────────
    chat_history: list[dict]            # [{"role": "user/assistant", "content": "..."}]

    # ── Control ───────────────────────────────────────────────
    error: str                          # error message if something fails