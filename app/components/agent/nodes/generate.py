"""
generate.py
-----------
LangGraph node: Generate

Responsible for synthesizing a final answer from retrieved documents
using the Groq LLM. Handles context building, source deduplication,
and inline citation formatting.

Flow position:
    grade_relevance → [rewrite_query] → generate → grade_hallucination
"""

from ..state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from app.components.agent.nodes.llm import get_llm


# ── Helpers ───────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication purposes.

    Strips 'www.' prefix and trailing slashes so that:
        https://www.pmjay.gov.in/about  ==  https://pmjay.gov.in/about/

    Args:
        url: Raw URL string from document payload

    Returns:
        Normalized URL string
    """
    return (
        url.replace("https://www.", "https://")
           .replace("http://www.", "http://")
           .rstrip("/")
    )


def build_context(docs: list[dict]) -> str:
    """
    Build a numbered context block from retrieved documents.

    Each document is formatted with its source title/URL, section
    breadcrumb, and chunk text. Documents are separated by a divider
    so the LLM can clearly distinguish between sources.

    Format per document:
        [1] Source: <title or url>
        URL: <url>
        Section: <heading breadcrumb>

        <chunk text>

    Args:
        docs: List of retrieved chunk dicts from Qdrant

    Returns:
        Single formatted string ready to inject into LLM prompt
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[{i}] Source: {doc.get('title') or doc.get('url')}\n"
            f"URL: {doc.get('url')}\n"
            f"Section: {doc.get('section', '')}\n\n"
            f"{doc['text']}"
        )
    return "\n\n---\n\n".join(parts)


def deduplicate_docs(docs: list[dict]) -> list[dict]:
    """
    Remove duplicate documents based on normalized URL.

    Duplicates arise when the same page is chunked into multiple
    overlapping pieces, or when the crawler indexes both the www.
    and non-www. variants of the same URL.

    Keeps the first occurrence (highest reranker score, since docs
    are already sorted by relevance before this node runs).

    Args:
        docs: List of retrieved chunk dicts, sorted by relevance score

    Returns:
        Deduplicated list preserving original order
    """
    seen_urls = set()
    unique = []
    for doc in docs:
        key = normalize_url(doc.get("url", ""))
        if key not in seen_urls:
            seen_urls.add(key)
            unique.append(doc)
    return unique


def format_sources(docs: list[dict]) -> list[dict]:
    """
    Convert raw chunk dicts into clean source citation records.

    Strips out heavy fields (chunk text, embeddings) and keeps only
    what is needed for display in the final response — index, URL,
    title, section, and relevance score.

    Args:
        docs: Deduplicated list of chunk dicts

    Returns:
        List of source citation dicts with keys:
            index, url, title, section, score
    """
    return [
        {
            "index":   i,
            "url":     doc.get("url", ""),
            "title":   doc.get("title", ""),
            "section": doc.get("section", ""),
            # Prefer reranker score if available, fall back to
            # retrieval score (RRF fusion score from Qdrant)
            "score":   doc.get("rerank_score", doc.get("retrieval_score", 0.0)),
        }
        for i, doc in enumerate(docs, 1)
    ]


# ── Node ──────────────────────────────────────────────────────────

def generate(state: AgentState) -> AgentState:
    """
    LangGraph node: Generate answer from retrieved documents.

    Uses relevant_docs if the relevance grader found sufficient docs,
    otherwise falls back to retrieved_docs (best-effort generation).
    Sends a structured prompt to Groq Llama-3.1-8B with the numbered
    context block and instructs it to cite sources inline.

    State reads:
        query         — original user question
        relevant_docs — docs that passed relevance grading
        retrieved_docs — fallback if relevant_docs is empty

    State writes:
        answer  — generated answer string with inline citations
        sources — list of deduplicated source citation dicts

    Args:
        state: Current AgentState

    Returns:
        Updated AgentState with answer and sources populated
    """
    query = state["query"]

    # Prefer relevance-graded docs; fall back to raw retrieved docs
    docs = state.get("relevant_docs") or state.get("retrieved_docs", [])

    # ── No docs found ─────────────────────────────────────────
    if not docs:
        return {
            **state,
            "answer": "I don't have enough information to answer this question.",
            "sources": []
        }

    # ── Build prompt ──────────────────────────────────────────
    unique_docs = deduplicate_docs(docs)
    context = build_context(unique_docs)

    system_prompt = """You are a helpful assistant. Answer the user's \
question using ONLY the information provided in the context below.

Rules:
- Cite sources inline using [1], [2] etc. after each claim
- If the answer is not in the context, say so clearly
- Be concise and factual
- Do not make up any information"""

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    # ── Generate ──────────────────────────────────────────────
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # ── Deduplicate and format sources ────────────────────────
    # unique_docs = deduplicate_docs(docs)
    sources     = format_sources(unique_docs)

    return {
        **state,
        "answer":  response.content.strip(),
        "sources": sources,
    }