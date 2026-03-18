"""
chunker.py
----------
Semantic chunker for crawled page JSONs.

Strategy:
- Split on sentence boundaries (not arbitrary char count)
- Every chunk carries full source metadata for citation
- Prepend heading context to each chunk so retrieval is section-aware
- Tables kept as atomic chunks (never split mid-table)
- Overlapping window to avoid losing context at boundaries

Output per chunk:
{
    "text": "...",          # content to embed
    "url": "...",
    "title": "...",
    "domain": "...",
    "depth": 0,
    "section": "About PM-JAY → Eligibility",   # heading breadcrumb
    "chunk_index": 0,
    "total_chunks": 5,
    "source_type": "webpage",  # webpage | table | meta
    "doc_id": "page_abc123",
}
"""

import re
import json
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from pathlib import Path
from app.config.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE

logger = get_logger(__name__)

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?।])\s+') ## to handle hindi purn viram

# ----------- Helpers --------------
def _split_sentences(text: str) -> list[str]:
    """
    split text into sentences. Handles Hindi and English
    """
    parts = SENTENCE_ENDINGS.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

def _build_chunks_from_sentences(
        sentences: list[str],
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
    """
    Greedily fill chunks upto chunk size chars.
    Add overlap by including Last N characters from previous chunks.
    Never split mid sentences
    """
    chunks = []
    current = []
    current_len = 0
    overlap_text = ""

    for sent in sentences:
        sent_len = len(sent)

        ## if single sentence is bigger than the chunk size, include it alone
        if sent_len > chunk_size and not current:
            ### if overlapping text is present add it to sentence and then add it to chunks
            if overlap_text:
                chunks.append((overlap_text + " " + sent).strip())
            else:
                chunks.append(sent)
            overlap_text = sent[-overlap:] ## skip overlap sized chunk from end
            continue
        
        if current_len + sent_len + 1 > chunk_size and current:
            chunk_text = ' '.join(current)
            ## if overlap text is present add it with chunk text for a full chunk
            full_chunk = (overlap_text + " " + chunk_text).strip() if overlap_text else chunk_text
            chunks.append(full_chunk)

            # overlap take tail of finished chunk
            overlap_text = chunk_text[-overlap] if len(chunk_text) > overlap else chunk_text
            
            ## empty current after adding it to chunks
            current = []
            current_len = 0
        
        current.append(sent)
        current_len += sent_len + 1
    
    # remaining current
    if current:
        chunk_text = ' '.join(current)
        full_chunk = (overlap_text + " " + chunk_text).strip() if overlap_text else chunk_text
        chunks.append(full_chunk)
    
    ## return list of chunks greater than min required chunk size
    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]

def _get_section_for_position(text: str, pos: int, headings: list[dict]) -> str:
    """
    Find which heading section a text position falls under
    returns breadcrumb string like 'About PMJay -> Eligibility'
    """
    if not headings:
        return ""
    
    # find headings which appear before this position in text
    for h in headings:
        h_text = h.get('text', '')
        h_pos = text.find(h_text)
        if 0 <= h_pos <= pos:
            best = h_text
    return best

def _heading_breadcrumb(heading_hierarchy: list[str], heading_text: str) -> str:
    """
    Find full breadcrumb path for a given heading text
    """
    for path in heading_hierarchy:
        if path.endswith(heading_text):
            return path
    return heading_text


# ******************** Chunker ************************ #
def chunk_page(page: dict) -> list[dict]:
    """
    Chunk a single crawled page JSON into RAG-ready chunks.
    Returns list of chunk dicts ready for embedding + Qdrant upload.
    """
    url      = page.get("url", "")
    title    = page.get("title", "")
    domain   = page.get("domain", "")
    depth    = page.get("depth", 0)
    doc_id   = page.get("doc_id", "")
    headings = page.get("headings", [])
    h_hier   = page.get("heading_hierarchy", [])
    body     = page.get("body_text", "").strip()
    tables   = page.get("tables", [])
    meta_desc = page.get("meta_description", "").strip()
 
    all_chunks = []
 
    # ── 1. Meta chunk ─────────────────────────────────────────────
    # Title + meta description as a standalone chunk
    # This ensures basic "what is this page about" queries hit something
    if title or meta_desc:
        meta_text = ""
        if title:
            meta_text += f"Page Title: {title}\n"
        if meta_desc:
            meta_text += f"Summary: {meta_desc}"
        if len(meta_text) >= MIN_CHUNK_SIZE:
            all_chunks.append({
                "text":        meta_text.strip(),
                "url":         url,
                "title":       title,
                "domain":      domain,
                "depth":       depth,
                "section":     "Page Summary",
                "chunk_index": 0,
                "source_type": "meta",
                "doc_id":      doc_id,
            })
 
    # ── 2. Body text chunks ───────────────────────────────────────
    if body:
        # Split body into sections by markdown headings
        # This gives us natural section boundaries
        sections = _split_by_headings(body)
 
        body_chunks = []
        for section_heading, section_text in sections:
            if not section_text.strip():
                continue
 
            sentences = _split_sentences(section_text)
            raw_chunks = _build_chunks_from_sentences(sentences)
 
            # Find breadcrumb for this section
            breadcrumb = _heading_breadcrumb(h_hier, section_heading) if section_heading else ""
 
            for chunk_text in raw_chunks:
                # Prepend heading context to chunk text
                # This makes the chunk self-contained for embedding
                if breadcrumb:
                    full_text = f"[Section: {breadcrumb}]\n\n{chunk_text}"
                else:
                    full_text = chunk_text
 
                body_chunks.append({
                    "text":        full_text,
                    "url":         url,
                    "title":       title,
                    "domain":      domain,
                    "depth":       depth,
                    "section":     breadcrumb or section_heading,
                    "chunk_index": 0,   # filled below
                    "source_type": "webpage",
                    "doc_id":      doc_id,
                })
 
        all_chunks.extend(body_chunks)
 
    # ── 3. Table chunks ───────────────────────────────────────────
    # Tables are kept atomic — never split across chunks
    for table in tables:
        md = table.get("markdown", "").strip()
        caption = table.get("caption", "")
        if not md or len(md) < MIN_CHUNK_SIZE:
            continue
 
        table_text = f"Table: {caption}\n\n{md}" if caption else md
        all_chunks.append({
            "text":        table_text,
            "url":         url,
            "title":       title,
            "domain":      domain,
            "depth":       depth,
            "section":     f"Table: {caption}" if caption else "Table",
            "chunk_index": 0,
            "source_type": "table",
            "doc_id":      doc_id,
        })
 
    # ── Fill chunk_index and total_chunks ─────────────────────────
    total = len(all_chunks)
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i
        chunk["total_chunks"] = total
 
    return all_chunks



def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Split markdown body text by heading lines (# ## ### etc.)
    Returns list of (heading_text, section_content) tuples.
    First tuple may have empty heading if content precedes first heading.
    """
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
 
    if not matches:
        return [("", text)]
 
    sections = []
 
    # Content before first heading
    if matches[0].start() > 0:
        sections.append(("", text[:matches[0].start()].strip()))
 
    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append((heading_text, content))
 
    return sections
 
 
# ── Batch processor ───────────────────────────────────────────────
 
def chunk_all_pages(pages_dir: str) -> list[dict]:
    """
    Load all page JSONs from output/pages/**/*.json
    Returns all chunks across all pages.
    """
    pages_path = Path(pages_dir)
    if not pages_path.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")
 
    all_chunks = []
    page_count = 0
    skipped = 0
 
    for json_file in pages_path.rglob("*.json"):
        try:
            page = json.loads(json_file.read_text(encoding="utf-8"))
 
            # Skip error pages
            if page.get("rag_status", {}).get("error"):
                skipped += 1
                continue
 
            # Skip pages with no body text
            if len(page.get("body_text", "")) < 100:
                skipped += 1
                continue
 
            chunks = chunk_page(page)
            all_chunks.extend(chunks)
            page_count += 1
 
        except Exception as e:
            logger.warning(f"Failed to chunk {json_file.name}: {e}")
            skipped += 1
 
    logger.info(
        f"Chunked {page_count} pages → {len(all_chunks)} chunks "
        f"({skipped} pages skipped)"
    )
    return all_chunks
 
# if __name__ == "__main__":
#     # Quick test
#     import sys
#     pages_dir = sys.argv[1] if len(sys.argv) > 1 else r"app\components\web_crawler\output\pages"
#     chunks = chunk_all_pages(pages_dir)
#     print(f"\nTotal chunks: {len(chunks)}")
#     print(f"\nSample chunk:")
#     if chunks:
#         c = chunks[10]
#         print(f"  source_type : {c['source_type']}")
#         print(f"  url         : {c['url']}")
#         print(f"  section     : {c['section']}")
#         print(f"  text[:200]  : {c['text']}")