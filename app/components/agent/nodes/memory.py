from ..state import AgentState

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