from ..state import AgentState
from langchain_core.messages import HumanMessage
from app.components.agent.nodes.llm import get_llm
from app.common.logger import get_logger

logger = get_logger(__name__)

def rewrite_query(state: AgentState) -> AgentState:
    """
    Rewrite the query when retrieved docs were not relevant.
    Uses LLM to rephrase — particularly useful for:
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
