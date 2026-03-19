"""
ingest.py
---------
Run this ONCE to ingest all crawled pages into Qdrant.

Usage:
    python ingest.py
    python ingest.py --pages ./output/pages --recreate

Steps:
    1. Load all page JSONs from output/pages/
    2. Chunk each page (semantic, sentence-aware)
    3. Embed all chunks 
    4. Upload to Qdrant Cloud with dense + sparse vectors
"""

import os
import sys
import time
from app.common.logger import get_logger
from app.components.rag_pipeline.ingestion.chunker import chunk_all_pages
from app.components.rag_pipeline.ingestion.embedder  import Embedder
from app.components.rag_pipeline.ingestion.qdrant_uploader import QdrantUploader
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


def run_ingestion(pages_dir: str, recreate: bool = False):

    start = time.time()
    print("\n" + "="*60)
    print("PHASE 2 — INGESTION PIPELINE")
    print("="*60)

    # ── Step 1: Chunk ──────────────────────────────────────────
    print("\n[1/4] Chunking pages...")
    chunks = chunk_all_pages(pages_dir)
    if not chunks:
        print("ERROR: No chunks produced. Check your pages directory.")
        return
    print(f"      {len(chunks)} chunks produced")

    # # Print sample
    # print(f"\n      Sample chunk:")
    # c = chunks[0]
    # print(f"        source_type : {c['source_type']}")
    # print(f"        section     : {c['section']}")
    # print(f"        text[:120]  : {c['text'][:120]}")

    # ── Step 2: Embed ──────────────────────────────────────────
    print("\n[2/4] Loading embedding model...")
    print("      (downloads once, then cached)")
    embedder = Embedder()

    print(f"\n[3/4] Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_passages(texts, show_progress=True)
    print(f"      Vectors shape: {vectors.shape}")

    # ── Step 3: Upload ─────────────────────────────────────────
    print("\n[4/4] Uploading to Qdrant Cloud...")
    uploader = QdrantUploader()
    uploader.create_collection(recreate=recreate)
    uploaded = uploader.upload_chunks(chunks, vectors)

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - start
    info = uploader.get_collection_info()

    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"  Chunks uploaded : {uploaded}")
    print(f"  Qdrant points   : {info['points']}")
    print(f"  Collection      : {info['name']}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print("="*60)
    print("\nReady for Phase 2 retrieval!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default=r"app\components\web_crawler\output\pages")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("QDRANT_URL"):
        print("ERROR: QDRANT_URL not set in .env file")
        print("  Add: QDRANT_URL=https://xxxx.us-east.aws.cloud.qdrant.io")
        sys.exit(1)

    run_ingestion(args.pages, recreate=args.recreate)


if __name__ == "__main__":
    main()