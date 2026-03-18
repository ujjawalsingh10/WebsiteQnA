from ..state import AgentState
from langchain_core.messages import HumanMessage
from app.components.agent.nodes.llm import get_llm
from app.common.logger import get_logger

logger = get_logger(__name__)

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