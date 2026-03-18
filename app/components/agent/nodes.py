"""
nodes.py
--------
Every node in the LangGraph agent.

Nodes:
  1. analyze_query      — detect language, intent, entities
  2. retrieve           — hybrid search + rerank from Qdrant
  3. grade_relevance    — are retrieved docs actually relevant?
  4. rewrite_query      — rewrite if docs were not relevant
  5. generate           — LLM synthesizes answer with citations
  6. grade_hallucination — is answer grounded in retrieved docs?
  7. grade_answer       — does answer actually address the question?
"""

import os
from app.common.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.components.agent.state import AgentState

# ── LLM setup ─────────────────────────────────────────────────────
def get_llm(temperature: float = 0.1) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        groq_api_key=api_key,
        max_tokens=512,
    )


# ─────────────────────────────────────────────────────────────────
# NODE 1 — Query Analyzer
# ─────────────────────────────────────────────────────────────────

def analyze_query(state: AgentState) -> AgentState:
    """
    Analyze the user query:
    - Detect language (en / hi / mixed)
    - Classify intent
    - Initialize rewrite counter
    """
    query = state["query"]

    # Simple language detection — check for Hindi Unicode range
    hindi_chars = sum(1 for c in query if '\u0900' <= c <= '\u097F')
    total_chars = len(query.replace(" ", ""))

    if total_chars == 0:
        language = "en"
    elif hindi_chars / total_chars > 0.5:
        language = "hi"
    elif hindi_chars > 0:
        language = "mixed"
    else:
        language = "en"

    # Intent classification using LLM
    llm = get_llm(temperature=0)
    prompt = f"""Classify this question about Indian government health scheme PM-JAY into one intent.
Reply with ONLY one word: factual | eligibility | procedural | comparison

Question: {query}
Intent:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        intent_raw = response.content.strip().lower()
        intent = intent_raw if intent_raw in ["factual", "eligibility", "procedural", "comparison"] else "factual"
    except Exception:
        intent = "factual"

    logger.info(f"Query analysis — language: {language}, intent: {intent}")

    return {
        **state,
        "language":       language,
        "intent":         intent,
        "rewritten_query": query,    # start with original
        "rewrite_count":  0,
        "chat_history":   state.get("chat_history", []),
    }


# ─────────────────────────────────────────────────────────────────
# NODE 2 — Retriever
# ─────────────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> AgentState:
    """
    Hybrid search from Qdrant using current query (original or rewritten).
    """
    from ..retrieval.retriever import get_retriever

    query = state.get("rewritten_query") or state["query"]
    retriever = get_retriever()

    try:
        docs = retriever.retrieve(query, top_k=5, candidate_count=20)
        logger.info(f"Retrieved {len(docs)} docs for query: {query[:60]}")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        docs = []

    return {**state, "retrieved_docs": docs}


# ─────────────────────────────────────────────────────────────────
# NODE 3 — Relevance Grader
# ─────────────────────────────────────────────────────────────────

def grade_relevance(state: AgentState) -> AgentState:
    """
    Grade each retrieved doc for relevance to the query.
    If fewer than 2 relevant docs → mark as insufficient → triggers rewrite.
    """
    query = state.get("rewritten_query") or state["query"]
    docs  = state.get("retrieved_docs", [])

    if not docs:
        return {**state, "relevant_docs": [], "relevance_grade": "insufficient"}

    llm = get_llm(temperature=0)
    relevant = []

    for doc in docs:
        prompt = f"""Is this document relevant to the question? Reply ONLY: yes | no

Question: {query}
Document: {doc['text'][:400]}
Answer:"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            if "yes" in response.content.strip().lower():
                relevant.append(doc)
        except Exception:
            relevant.append(doc)   # assume relevant on error

    grade = "sufficient" if len(relevant) >= 2 else "insufficient"
    logger.info(f"Relevance grading — {len(relevant)}/{len(docs)} relevant, grade: {grade}")

    return {**state, "relevant_docs": relevant, "relevance_grade": grade}


# ─────────────────────────────────────────────────────────────────
# NODE 4 — Query Rewriter
# ─────────────────────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> AgentState:
    """
    Rewrite the query when retrieved docs were not relevant.
    Uses LLM to rephrase — particularly useful for:
    - Hinglish queries → cleaner English
    - Vague queries → more specific
    - Acronyms → expanded form
    Max 2 rewrites to avoid infinite loops.
    """
    original  = state["query"]
    current   = state.get("rewritten_query", original)
    count     = state.get("rewrite_count", 0)
    chat_hist = state.get("chat_history", [])

    llm = get_llm(temperature=0.3)

    # Include recent chat context in rewrite prompt
    context = ""
    if chat_hist:
        last = chat_hist[-2:]  # last 1 exchange
        context = "\n".join([f"{m['role']}: {m['content']}" for m in last])
        context = f"\nPrevious conversation:\n{context}\n"

    prompt = f"""Rewrite this question to improve search results from a government health scheme database.
- Expand abbreviations (PMJAY → Pradhan Mantri Jan Arogya Yojana)
- Translate Hindi/Hinglish to English if needed
- Make it more specific
- Keep it concise{context}

Original question: {original}
Current question: {current}

Rewritten question (one line only):"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        new_query = response.content.strip().split("\n")[0].strip()
        # Remove quotes if LLM added them
        new_query = new_query.strip('"\'')
    except Exception:
        new_query = current

    logger.info(f"Query rewrite #{count + 1}: '{current}' → '{new_query}'")

    return {
        **state,
        "rewritten_query": new_query,
        "rewrite_count": count + 1,
    }


# ─────────────────────────────────────────────────────────────────
# NODE 5 — Generator
# ─────────────────────────────────────────────────────────────────

def generate(state: AgentState) -> AgentState:
    """
    Generate final answer using retrieved docs as context.
    Cites sources inline. Responds in same language as query.
    """
    query     = state["query"]
    language  = state.get("language", "en")
    docs      = state.get("relevant_docs") or state.get("retrieved_docs", [])
    chat_hist = state.get("chat_history", [])

    if not docs:
        answer = (
            "मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"
            if language == "hi"
            else "I don't have enough information to answer this question."
        )
        return {**state, "answer": answer, "sources": []}

    # Build context block
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"[{i}] Source: {doc.get('title') or doc.get('url')}\n"
            f"URL: {doc.get('url')}\n"
            f"Section: {doc.get('section', '')}\n\n"
            f"{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history context
    history_text = ""
    if chat_hist:
        history_text = "\n\nConversation history:\n"
        for msg in chat_hist[-4:]:  # last 2 exchanges
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    lang_instruction = (
        "Respond in Hindi if the question is in Hindi. "
        "Respond in English if the question is in English. "
        "For mixed queries, respond in English."
    )

    system_prompt = f"""You are a helpful assistant for PM-JAY (Pradhan Mantri Jan Arogya Yojana), India's government health insurance scheme.

Answer questions using ONLY the provided context. Rules:
- Use only information from the context below
- If the answer is not in the context, say so clearly
- Cite sources using [1], [2] etc. after each claim
- Be concise and factual
- {lang_instruction}
- Do not make up eligibility criteria, benefit amounts, or policy details"""

    user_prompt = f"""Context:
{context}
{history_text}
Question: {query}

Answer:"""

    llm = get_llm(temperature=0.1)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = "Sorry, I encountered an error generating the answer. Please try again."

    sources = [
        {
            "index":       i + 1,
            "url":         doc.get("url", ""),
            "title":       doc.get("title", ""),
            "section":     doc.get("section", ""),
            "score":       doc.get("rerank_score", doc.get("retrieval_score", 0.0)),
        }
        for i, doc in enumerate(docs)
    ]

    logger.info(f"Generated answer ({len(answer)} chars) from {len(docs)} docs")

    return {**state, "answer": answer, "sources": sources}


# ─────────────────────────────────────────────────────────────────
# NODE 6 — Hallucination Grader
# ─────────────────────────────────────────────────────────────────

def grade_hallucination(state: AgentState) -> AgentState:
    """
    Check if the answer is grounded in retrieved documents.
    If hallucinated → triggers regeneration (once).
    """
    answer = state.get("answer", "")
    docs   = state.get("relevant_docs") or state.get("retrieved_docs", [])

    if not answer or not docs:
        return {**state, "hallucination_grade": "grounded"}

    context = "\n\n".join([d["text"][:300] for d in docs[:3]])

    llm = get_llm(temperature=0)
    prompt = f"""Is this answer fully supported by the provided context?
Reply ONLY: grounded | hallucinated

Context:
{context}

Answer: {answer[:500]}

Grade:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        grade_raw = response.content.strip().lower()
        grade = "hallucinated" if "hallucinated" in grade_raw else "grounded"
    except Exception:
        grade = "grounded"

    logger.info(f"Hallucination grade: {grade}")
    return {**state, "hallucination_grade": grade}


# ─────────────────────────────────────────────────────────────────
# NODE 7 — Answer Grader
# ─────────────────────────────────────────────────────────────────

def grade_answer(state: AgentState) -> AgentState:
    """
    Check if the answer actually addresses the user's question.
    """
    query  = state["query"]
    answer = state.get("answer", "")

    if not answer:
        return {**state, "answer_grade": "does_not_answer"}

    llm = get_llm(temperature=0)
    prompt = f"""Does this answer address the question asked?
Reply ONLY: answers_question | does_not_answer

Question: {query}
Answer: {answer[:400]}

Grade:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        grade_raw = response.content.strip().lower()
        grade = "answers_question" if "answers_question" in grade_raw else "does_not_answer"
    except Exception:
        grade = "answers_question"

    logger.info(f"Answer grade: {grade}")
    return {**state, "answer_grade": grade}


# ─────────────────────────────────────────────────────────────────
# NODE 8 — Memory Updater
# ─────────────────────────────────────────────────────────────────

def update_memory(state: AgentState) -> AgentState:
    """
    Append current Q&A to conversation history for multi-turn memory.
    """
    chat_hist = list(state.get("chat_history", []))
    chat_hist.append({"role": "user",      "content": state["query"]})
    chat_hist.append({"role": "assistant", "content": state.get("answer", "")})

    # Keep last 10 exchanges (20 messages) to avoid context overflow
    if len(chat_hist) > 20:
        chat_hist = chat_hist[-20:]

    return {**state, "chat_history": chat_hist}