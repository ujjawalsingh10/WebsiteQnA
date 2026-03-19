from ..state import AgentState
from langchain_core.messages import HumanMessage
from app.components.agent.nodes.llm import get_llm
from app.common.logger import get_logger

logger = get_logger(__name__)

def grade_relevance(state: AgentState) -> AgentState:
    """
    Grade each retrieved doc for relevance to the query.
    If not enough retrieved → mark as insufficient → triggers rewrite.
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

    grade = "sufficient" if len(relevant) >= 1 else "insufficient"
    logger.info(f"Relevance grading — {len(relevant)}/{len(docs)} relevant, grade: {grade}")

    return {**state, "relevant_docs": relevant, "relevance_grade": grade}
