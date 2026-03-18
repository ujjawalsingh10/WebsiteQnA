"""
qdrant_uploader.py
------------------
Creates Qdrant collection and uploads chunks with:
- Dense vectors 
- Sparse vectors (BM25-style for hybrid search)
- Full metadata as payload (url, title, section, etc.)

Hybrid search = dense similarity + keyword matching
Critical for exact string matching, not just semantic concepts.
"""

import os
import uuid
import logging
from dotenv import load_dotenv
from app.common.logger import get_logger
from app.config.config import COLLECTION_NAME, DENSE_VECTOR_DIM, QDRANT_BATCH_SIZE

load_dotenv()

logger = get_logger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams,
        PointStruct, SparseVector,
        SparseVectorParams, SparseIndexParams,
        HnswConfigDiff, OptimizersConfigDiff,
        models as qmodels,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.error("qdrant-client not installed: pip install qdrant-client")

try:
    from qdrant_client.models import SparseVectorParams, SparseIndexParams
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False


class QdrantUploader:
    """
    Manages the Qdrant collection and chunk uploads.

    Collection has:
    - "dense"  : embedding vectors (cosine)
    - "sparse" : BM25-style sparse vectors (dot product) for keyword matching
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")

        self.url = url or os.environ.get("QDRANT_URL", "")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY", "")
        self.collection_name = collection_name

        if not self.url:
            raise ValueError(
                "QDRANT_URL not set. Add to .env file:\n"
                "QDRANT_URL=https://xxxx.us-east.aws.cloud.qdrant.io"
            )

        logger.info(f"Connecting to Qdrant: {self.url}")
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            verify = False,
            timeout=60,
        )
        logger.info("Qdrant connected")

    def create_collection(self, recreate: bool = False):
        """
        Create the Qdrant collection with dense + sparse vectors.
        If recreate=True, drops existing collection first (fresh ingestion).
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and not recreate:
            info = self.client.get_collection(self.collection_name)
            count = info.points_count or 0
            logger.info(
                f"Collection '{self.collection_name}' already exists "
                f"({count} points) — skipping creation"
            )
            return

        if exists and recreate:
            logger.info(f"Dropping existing collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)

        logger.info(f"Creating collection '{self.collection_name}'")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_VECTOR_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,              # HNSW graph connections
                        ef_construct=100,  # build-time accuracy
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            } if SPARSE_AVAILABLE else {},
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,  # index immediately (good for small collections)
            ),
        )
        logger.info(f"Collection '{self.collection_name}' created")

    def _text_to_sparse(self, text: str) -> dict:
        """
        Simple term-frequency sparse vector for BM25-style keyword search.
        To do -  use proper BM25 encoder, 
        this is a lightweight approximation sufficient for hybrid search.
        """
        import hashlib

        words = text.lower().split()
        tf: dict[int, float] = {}
        for word in words:
            # Use hash of word as token ID (avoids needing a vocabulary)
            token_id = int(hashlib.md5(word.encode()).hexdigest(), 16) % 100_000
            tf[token_id] = tf.get(token_id, 0) + 1.0

        if not tf:
            return {"indices": [0], "values": [0.0]}

        # Normalize
        max_tf = max(tf.values())
        return {
            "indices": list(tf.keys()),
            "values":  [v / max_tf for v in tf.values()],
        }

    def upload_chunks(
        self,
        chunks: list[dict],
        dense_vectors: "np.ndarray",
        show_progress: bool = True,
    ) -> int:
        """
        Upload chunks with their dense + sparse vectors to Qdrant.

        Args:
            chunks: list of chunk dicts from chunker.py
            dense_vectors: numpy array of shape (N, embed_dim)
            show_progress: print progress

        Returns:
            number of points uploaded
        """
        import numpy as np

        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(dense_vectors)}) "
                "must have same length"
            )

        total = len(chunks)
        uploaded = 0

        for batch_start in range(0, total, QDRANT_BATCH_SIZE):
            batch_end   = min(batch_start + QDRANT_BATCH_SIZE, total)
            batch_chunks = chunks[batch_start:batch_end]
            batch_vecs   = dense_vectors[batch_start:batch_end]

            points = []
            for chunk, dense_vec in zip(batch_chunks, batch_vecs):
                sparse = self._text_to_sparse(chunk["text"])

                # Build payload — everything stored here is retrievable
                # without any additional DB lookup
                payload = {
                    "text":         chunk["text"],
                    "url":          chunk["url"],
                    "title":        chunk["title"],
                    "domain":       chunk["domain"],
                    "depth":        chunk["depth"],
                    "section":      chunk.get("section", ""),
                    "chunk_index":  chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "source_type":  chunk.get("source_type", "webpage"),
                    "doc_id":       chunk.get("doc_id", ""),
                }

                vectors = {"dense": dense_vec.tolist()}
                if SPARSE_AVAILABLE:
                    vectors["sparse"] = SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    )

                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors,
                    payload=payload,
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            uploaded += len(points)

            if show_progress:
                pct = uploaded / total * 100
                print(f"  Uploaded {uploaded}/{total} chunks ({pct:.0f}%)...", end="\r")

        if show_progress:
            print(f"  Uploaded {uploaded}/{total} chunks (100%) ✓          ")

        return uploaded

    def get_collection_info(self) -> dict:
        """Return collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name":   self.collection_name,
            "points": info.points_count,
            "status": str(info.status),
        }