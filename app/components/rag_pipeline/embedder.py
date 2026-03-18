"""
embedder.py
-----------
Embeds chunks using Embedding model.
"""

import logging
import numpy as np
from app.common.logger import get_logger
from app.config.config import HUGGINGFACE_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from typing import Union
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# ── Config ─────────────────────────────────────────────

# Example value in config:
# HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

EMBED_MODEL = HUGGINGFACE_EMBEDDING_MODEL
BATCH_SIZE = EMBEDDING_BATCH_SIZE


class Embedder:
    """
    Wrapper around SentenceTransformer for embedding documents and queries.

    Responsibilities:
    - Load embedding model once
    - Convert text into dense vectors
    - Normalize embeddings for cosine similarity search

    Normalized embeddings allow cosine similarity to be computed
    using a simple dot product in vector databases such as Qdrant.
    """

    def __init__(self, model_name: str = EMBED_MODEL):

        logger.info(f"Loading embedding model: {model_name}")
        logger.info("(First run downloads model and caches locally)")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # automatically determine embedding dimension
        self.dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded | dim={self.dim}")


    # ── Document Embedding ─────────────────────────────

    def embed_passages(
        self,
        texts: list[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed document chunks before storing them in the vector database.

        Parameters
        ----------
        texts : list[str]
            List of chunk texts extracted from documents.

        show_progress : bool
            Whether to display encoding progress bar.

        Returns
        -------
        np.ndarray
            Matrix of embeddings with shape:
            (num_chunks, embedding_dimension)
        """

        vectors = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return vectors.astype(np.float32)


    # ── Query Embedding ─────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query for semantic search.

        Parameters
        ----------
        query : str
            User's question or search query.

        Returns
        -------
        np.ndarray
            Vector representation of the query.
        """

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        return vector.astype(np.float32)


    # ── Batch Query Embedding ─────────────────────────

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """
        Embed multiple queries in batch.

        Useful for evaluation pipelines or testing retrieval performance.
        """

        vectors = self.model.encode(
            queries,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return vectors.astype(np.float32)


# ── Singleton Helper ─────────────────────────────────
# Ensures the embedding model loads only once across the application

_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    """
    Get or create a global embedder instance.

    This prevents repeatedly loading the embedding model
    which would waste memory and increase startup time.
    """

    global _embedder_instance

    if _embedder_instance is None:
        _embedder_instance = Embedder()

    return _embedder_instance


# # ── Quick Test ───────────────────────────────────────

# if __name__ == "__main__":

#     embedder = Embedder()

#     # Example query
#     query = "Who is eligible for PM-JAY?"

#     query_vec = embedder.embed_query(query)

#     print(f"Query vector shape: {query_vec.shape}")
#     print(f"Vector norm: {np.linalg.norm(query_vec):.4f}")

#     # Example document passages
#     passages = [
#         "PM-JAY provides health insurance coverage to economically vulnerable families.",
#         "The scheme offers cashless treatment at empanelled hospitals across India."
#     ]

#     vecs = embedder.embed_passages(passages)

#     print(f"Passage vectors shape: {vecs.shape}")

#     # similarity test
#     similarity = float(query_vec @ vecs[0])
#     print(f"Similarity score: {similarity:.4f}")