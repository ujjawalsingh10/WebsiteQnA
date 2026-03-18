"""
main.py
-------
Run the Phase 2 RAG agent interactively.

Usage:
    cd app/components/rag_pipeline
    python main.py

Requires .env with:
    QDRANT_URL=https://xxxx.us-east.aws.cloud.qdrant.io
    QDRANT_API_KEY=eyJ...
    GROQ_API_KEY=gsk_...
"""

import os
import sys
import textwrap
from app.common.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

# ── Env checks ────────────────────────────────────────────────────
def check_env():
    missing = []
    for key in ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]:
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print("ERROR: Missing environment variables:")
        for k in missing:
            print(f"  {k} — add to .env file")
        sys.exit(1)

# ── Pretty print ──────────────────────────────────────────────────
def print_result(result: dict):
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    answer = result.get("answer", "No answer generated.")
    for line in answer.split("\n"):
        print(textwrap.fill(line, width=70) if line.strip() else "")

    sources = result.get("sources", [])
    if sources:
        print("\n" + "-"*60)
        print("SOURCES")
        print("-"*60)
        for src in sources:
            score = src.get("score", 0)
            print(f"[{src['index']}] {src['url']}")
            if src.get("section"):
                print(f"     section : {src['section']}")
            print(f"     score   : {score:.3f}")

    # Show agent trace info
    # lang    = result.get("language", "")
    # intent  = result.get("intent", "")
    rewrites = result.get("rewrite_count", 0)
    hall    = result.get("hallucination_grade", "")
    rewritten = result.get("rewritten_query", "")
    original  = result.get("query", "")

    print("\n" + "-"*60)
    print("AGENT TRACE")
    print("-"*60)
    # print(f"  Language    : {lang}")
    # print(f"  Intent      : {intent}")
    if rewritten != original:
        print(f"  Rewrote to  : {rewritten}")
        print(f"  Rewrites    : {rewrites}")
    print(f"  Hallucination check : {hall}")
    print("="*60 + "\n")


# ── Main loop ─────────────────────────────────────────────────────
def main():
    check_env()

    print("\n" + "="*60)
    print("PM-JAY RAG AGENT — Phase 2")
    print("="*60)
    print("Loading models (first run takes ~60s)...")

    from app.components.agent.agent_graph  import get_graph
    graph = get_graph()

    print("Ready!\n")
    print("Type your question in English or Hindi.")
    print("Type 'quit' to exit.\n")

    # State persists across turns for conversation memory
    chat_history = []

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        try:
            result = graph.invoke({
                "query":        query,
                "chat_history": chat_history,
                "rewrite_count": 0,
            })

            print_result(result)

            # Persist chat history across turns
            chat_history = result.get("chat_history", [])

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()