from ..state import AgentState

def analyze_query(state: AgentState) -> AgentState:
    """
    Initializes query rewrite state
    """
    return {
        **state,
        "rewritten_query": state["query"],
        "rewrite_count": 0,
        "chat_history": state.get("chat_history", []),
    }

