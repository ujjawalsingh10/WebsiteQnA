"""
retriever.py
------------
Hybrid retrieval from Qdrant + cross-encoder reranking.

Two-stage retrieval:
  Stage 1: Hybrid search (dense + sparse) → top-20 candidates
  Stage 2: Cross-encoder reranker scores all 20 → returns top-k

Why two stages?
  Dense search is fast but imprecise at top-k.
  Cross-encoder is precise but slow — can't run on whole collection.
  Together: fast broad recall + precise final ranking.
"""

import os
import logging
from dotenv import load_dotenv
from app.config.config import COLLECTION_NAME, RERANKER_MODEL, RERANK_TOP_K, MIN_SCORE, CANDIDATE_COUNT
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SearchRequest, NamedVector, NamedSparseVector,
    SparseVector, FusionQuery, Prefetch, Query,
    models,
)
from app.components.rag_pipeline.ingestion.embedder import get_embedder

load_dotenv()
logger = logging.getLogger(__name__)


try:
    from sentence_transformers import CrossEncoder
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False

class Retriever:
    """
    Hybrid retriever with reranking.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = COLLECTION_NAME,
        use_reranker: bool = True,
    ):

        self.url = url or os.environ.get("QDRANT_URL", "")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY", "")
        self.collection_name = collection_name
        self.use_reranker = use_reranker and CE_AVAILABLE

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30,
        )

        self.embedder = get_embedder()

        if self.use_reranker:
            logger.info(f"Loading reranker: {RERANKER_MODEL}")
            self.reranker = CrossEncoder(RERANKER_MODEL)
            logger.info("Reranker loaded")
        else:
            self.reranker = None
            if not CE_AVAILABLE:
                logger.warning(
                    "CrossEncoder not available — retrieval without reranking. "
                    "Install: pip install sentence-transformers"
                )

    def _text_to_sparse(self, text: str) -> SparseVector:
        """Same sparse encoding as uploader."""
        import hashlib
        words = text.lower().split()
        tf: dict[int, float] = {}
        for word in words:
            token_id = int(hashlib.md5(word.encode()).hexdigest(), 16) % 100_000
            tf[token_id] = tf.get(token_id, 0) + 1.0
        if not tf:
            return SparseVector(indices=[0], values=[0.0])
        max_tf = max(tf.values())
        return SparseVector(
            indices=list(tf.keys()),
            values=[v / max_tf for v in tf.values()],
        )

    def retrieve(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        candidate_count: int = CANDIDATE_COUNT,
        filter_domain: str | None = None,
        filter_source_type: str | None = None,
    ) -> list[dict]:
        """
        Main retrieval method.

        Args:
            query: user question (English or Hindi)
            top_k: number of results to return after reranking
            candidate_count: number of candidates before reranking
            filter_domain: only retrieve from this domain
            filter_source_type: "webpage" | "table" | "meta" | "pdf_page"

        Returns:
            list of result dicts sorted by relevance score
        """

        # 1. Embed query
        query_vec = self.embedder.embed_query(query)
        sparse_vec = self._text_to_sparse(query)

        # 2. Build optional filter
        qdrant_filter = self._build_filter(filter_domain, filter_source_type)

        # 3. Hybrid search — dense + sparse with RRF fusion
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Dense retrieval
                    Prefetch(
                        query=query_vec.tolist(),
                        using="dense",
                        limit=candidate_count,
                        filter=qdrant_filter,
                    ),
                    # Sparse retrieval
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                        using="sparse",
                        limit=candidate_count,
                        filter=qdrant_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=candidate_count,
                with_payload=True,
            )
            hits = results.points

        except Exception as e:
            # Fallback to dense-only if hybrid fails
            logger.warning(f"Hybrid search failed ({e}), falling back to dense-only")
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vec.tolist()),
                limit=candidate_count,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            hits = results

        if not hits:
            return []

        # 4. Build candidate list
        candidates = []
        for hit in hits:
            payload = hit.payload or {}
            score = hit.score if hasattr(hit, 'score') else 0.0
            candidates.append({
                "text":         payload.get("text", ""),
                "url":          payload.get("url", ""),
                "title":        payload.get("title", ""),
                "domain":       payload.get("domain", ""),
                "section":      payload.get("section", ""),
                "source_type":  payload.get("source_type", "webpage"),
                "doc_id":       payload.get("doc_id", ""),
                "chunk_index":  payload.get("chunk_index", 0),
                "retrieval_score": float(score),
            })

        # Filter out low-score candidates
        candidates = [c for c in candidates if c["retrieval_score"] >= MIN_SCORE]

        # 5. Rerank
        if self.use_reranker and self.reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Re-score candidates using cross-encoder.
        Cross-encoder reads query+passage together → much more accurate than bi-encoder.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _build_filter(
        self,
        domain: str | None,
        source_type: str | None,
    ):
        """Build Qdrant filter from optional parameters."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = []
        if domain:
            conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        if source_type:
            conditions.append(
                FieldCondition(key="source_type", match=MatchValue(value=source_type))
            )

        if not conditions:
            return None

        return Filter(must=conditions)


# ── Singleton ─────────────────────────────────────────────────────
_retriever_instance: Retriever | None = None

def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance