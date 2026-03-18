from ..state import AgentState
from langchain_core.messages import HumanMessage
from app.components.agent.nodes.llm import get_llm
from app.common.logger import get_logger

logger = get_logger(__name__)

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
