"""
test_rag.py
-----------
Quick RAG test on crawled text.
No vector DB — uses in-memory numpy for similarity search.
Uses HuggingFace for embeddings + Llama for generation.

Usage:
    python test_rag.py
    python test_rag.py --output ./output --top-k 5

Requirements:
    pip install sentence-transformers langchain-huggingface huggingface-hub numpy python-dotenv

.env file:
    HUGGINGFACEHUB_API_TOKEN=hf_...
"""

import os
import json
import argparse
import textwrap
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

EMBED_MODEL   = "sentence-transformers/all-mpnet-base-v2"
CHAT_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
CHUNK_SIZE    = 400     # tokens approx (1 token ~ 4 chars)
CHUNK_OVERLAP = 80      # chars overlap between chunks
CHARS_PER_TOK = 4

# ─────────────────────────────────────────────────────────────────
# Init models (done once at startup)
# ─────────────────────────────────────────────────────────────────

def load_models():
    print("  Loading embedding model (downloads once, cached after)...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("  Connecting to LLM via HuggingFace Inference API...")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=CHAT_MODEL,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    return embedder, llm


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load pages
# ─────────────────────────────────────────────────────────────────

def load_pages(output_dir: str) -> list[dict]:
    pages = []
    pages_dir = Path(output_dir) / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"No pages directory found at: {pages_dir}")

    for json_file in pages_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            text = data.get("body_text", "").strip()
            if len(text) < 100:
                continue
            pages.append({
                "url":     data.get("url", ""),
                "title":   data.get("title", ""),
                "text":    text,
                "section": " -> ".join(data.get("heading_hierarchy", [])[:3]),
            })
        except Exception as e:
            print(f"  Skipping {json_file.name}: {e}")

    print(f"  Loaded {len(pages)} pages with usable text")
    return pages


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Chunk
# ─────────────────────────────────────────────────────────────────

def chunk_text(text, url, title, section):
    chunk_chars   = CHUNK_SIZE * CHARS_PER_TOK
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOK
    chunks, start = [], 0
    while start < len(text):
        body = text[start : start + chunk_chars].strip()
        if len(body) > 50:
            chunks.append({"text": body, "url": url, "title": title, "section": section})
        start += chunk_chars - overlap_chars
    return chunks

def build_chunks(pages):
    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_text(p["text"], p["url"], p["title"], p["section"]))
    print(f"  Built {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Embed
# ─────────────────────────────────────────────────────────────────

def embed_texts(texts, embedder):
    print(f"  Embedding {len(texts)} chunks locally...")
    vecs = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Retrieve
# ─────────────────────────────────────────────────────────────────

def retrieve(query, chunks, corpus_vecs, embedder, top_k=5):
    query_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)
    scores  = corpus_vecs @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx]


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Generate
# ─────────────────────────────────────────────────────────────────

def build_context(retrieved):
    parts = []
    for i, c in enumerate(retrieved, 1):
        parts.append(
            f"[{i}] Source: {c['title'] or c['url']}\n"
            f"    URL: {c['url']}\n"
            f"    Section: {c['section']}\n\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def ask(query, chunks, corpus_vecs, embedder, llm, top_k=5):
    retrieved = retrieve(query, chunks, corpus_vecs, embedder, top_k)
    context   = build_context(retrieved)

    system_prompt = """You are a helpful assistant answering questions using ONLY the provided context.
Rules:
- Use only information from the context.
- If not in context, say "I don't have enough information to answer this."
- Cite source [number] and URL for every claim.
- Be concise and factual."""

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {"answer": response.content.strip(), "sources": retrieved}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def print_answer(result):
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    for line in result["answer"].split("\n"):
        print(textwrap.fill(line, width=70) if line.strip() else "")
    print("\n" + "-" * 60)
    print("SOURCES RETRIEVED")
    print("-" * 60)
    for i, src in enumerate(result["sources"], 1):
        print(f"[{i}] score={src['score']:.3f}  {src['url']}")
        if src.get("section"):
            print(f"     section: {src['section']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./output")
    parser.add_argument("--top-k",  type=int, default=5)
    args = parser.parse_args()

    # if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    #     print("ERROR: HUGGINGFACEHUB_API_TOKEN not set.")
    #     print("  Add to .env:  HUGGINGFACEHUB_API_TOKEN=hf_...")
    #     return

    print("\n── Loading models ────────────────────────────────────")
    embedder, llm = load_models()

    print("\n── Loading pages ─────────────────────────────────────")
    pages = load_pages(args.output)

    print("\n── Chunking ──────────────────────────────────────────")
    chunks = build_chunks(pages)

    print("\n── Embedding ─────────────────────────────────────────")
    corpus_vecs = embed_texts([c["text"] for c in chunks], embedder)

    print("\n── Ready! Type your question (or 'quit' to exit) ─────\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!"); break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!"); break

        try:
            result = ask(query, chunks, corpus_vecs, embedder, llm, top_k=args.top_k)
            print_answer(result)
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()