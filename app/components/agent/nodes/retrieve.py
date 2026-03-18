from ..state import AgentState
from app.common.logger import get_logger

logger = get_logger(__name__)

def retrieve(state: AgentState) -> AgentState:
    """
    Hybrid search from Qdrant using current query (original or rewritten).
    """
    from app.components.rag_pipeline.retrieval.retriever import get_retriever

    query = state.get("rewritten_query") or state["query"]
    retriever = get_retriever()

    try:
        docs = retriever.retrieve(query, top_k=5, candidate_count=20)
        logger.info(f"Retrieved {len(docs)} docs for query: {query[:60]}")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        docs = []

    return {**state, "retrieved_docs": docs}