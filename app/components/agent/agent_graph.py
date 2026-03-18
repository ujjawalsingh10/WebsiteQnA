"""
graph.py
--------
LangGraph state machine definition.

Flow:
  analyze_query
      │
      ▼
  retrieve
      │
      ▼
  grade_relevance ──── insufficient (& rewrite_count < 2) ──► rewrite_query ──► retrieve
      │                                                                              (loop)
      │ sufficient
      ▼
  generate
      │
      ▼
  grade_hallucination ── hallucinated (once) ──► generate
      │                                           (loop)
      │ grounded
      ▼
  grade_answer
      │
      ▼
  update_memory
      │
      ▼
     END
"""

from app.common.logger import get_logger
from langgraph.graph import StateGraph, END

from .state import AgentState
from app.components.agent.nodes.analyze_query import analyze_query
from app.components.agent.nodes.retrieve import retrieve
from app.components.agent.nodes.relevance_grader import grade_relevance
from app.components.agent.nodes.rewrite_query import rewrite_query
from app.components.agent.nodes.generate import generate
from app.components.agent.nodes.hallucination_review import grade_hallucination
from app.components.agent.nodes.answer_grade import grade_answer
from app.components.agent.nodes.memory import update_memory

logger = get_logger(__name__)

# ── Conditional edge functions ────────────────────────────────────

def route_after_relevance(state: AgentState) -> str:
    """
    After grading relevance:
    - If sufficient docs → generate answer
    - If insufficient AND haven't rewritten twice → rewrite query
    - If insufficient AND already rewritten twice → generate anyway (best effort)
    """
    grade         = state.get("relevance_grade", "sufficient")
    rewrite_count = state.get("rewrite_count", 0)

    if grade == "sufficient":
        return "generate"
    elif rewrite_count < 2:
        logger.info(f"Insufficient docs — rewriting query (attempt {rewrite_count + 1})")
        return "rewrite_query"
    else:
        logger.info("Insufficient docs after 2 rewrites — generating best-effort answer")
        return "generate"


def route_after_hallucination(state: AgentState) -> str:
    """
    After hallucination check:
    - grounded → proceed to answer grader
    - hallucinated → regenerate (only once to avoid infinite loop)
    """
    grade         = state.get("hallucination_grade", "grounded")
    rewrite_count = state.get("rewrite_count", 0)   # reuse as regen counter

    if grade == "grounded":
        return "grade_answer"
    elif rewrite_count < 3:   # allow one regeneration attempt
        logger.info("Hallucination detected — regenerating answer")
        return "generate"
    else:
        logger.info("Hallucination persists — proceeding with current answer")
        return "grade_answer"


def route_after_answer_grade(state: AgentState) -> str:
    """Always proceed to memory update regardless of answer grade."""
    return "update_memory"


# ── Build graph ───────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph agent.
    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze_query",        analyze_query)
    graph.add_node("retrieve",             retrieve)
    graph.add_node("grade_relevance",      grade_relevance)
    graph.add_node("rewrite_query",        rewrite_query)
    graph.add_node("generate",             generate)
    graph.add_node("grade_hallucination",  grade_hallucination)
    graph.add_node("grade_answer",         grade_answer)
    graph.add_node("update_memory",        update_memory)

    # Entry point
    graph.set_entry_point("analyze_query")

    # Fixed edges
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve",      "grade_relevance")
    graph.add_edge("rewrite_query", "retrieve")         # loop back after rewrite

    # Conditional edges
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_relevance,
        {
            "generate":      "generate",
            "rewrite_query": "rewrite_query",
        }
    )

    graph.add_conditional_edges(
        "grade_hallucination",
        route_after_hallucination,
        {
            "grade_answer": "grade_answer",
            "generate":     "generate",
        }
    )

    graph.add_edge("generate",      "grade_hallucination")
    graph.add_edge("grade_answer",  "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()


# ── Singleton ─────────────────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def visualize_graph():
    graph = get_graph()
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

if __name__ == "__main__":
    visualize_graph()