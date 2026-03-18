from ..state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from app.components.agent.nodes.llm import get_llm

def generate(state: AgentState) -> AgentState:
    query = state["query"]
    docs = state.get("relevant_docs") or state.get("retrieved_docs", [])

    if not docs:
        return {
            **state,
            "answer": "I don't have enough information.",
            "sources": []
        }

    # context = "\n\n".join([d["text"] for d in docs])
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


    system = """Answer using ONLY the context. Cite sources."""
    user = f"Context:\n{context}\n\nQuestion: {query}"

    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])

    return {
    **state,
    "answer": response.content.strip(),
    "sources": [
        {
            "index":   i,
            "url":     doc.get("url", ""),
            "title":   doc.get("title", ""),
            "section": doc.get("section", ""),
            "score":   doc.get("rerank_score", doc.get("retrieval_score", 0.0)),
        }
        for i, doc in enumerate(docs, 1)
    ]
}