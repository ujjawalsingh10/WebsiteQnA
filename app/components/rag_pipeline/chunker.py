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
            overlap_text = sent[-overlap:] ## overlap sized chunk from end
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
    Chunk a single crawled json page into a RAG ready chunks.
    Returns a list of chunk dict ready for embedding + Qdrant upload
    """
    url = page.get('url', '')
    title = page.get('title', '')
    domain = page.get('domain', '')
    depth = page.get('depth', '')
    doc_id = page.get('doc_id', '')
    headings = page.get('headings', '')
    h_tier = page.get('heading_hierarchy', [])
    body = page.get('body_text', '')
    tables = page.get('tables', '')
    meta_desc = page.get('meta_description', '').strip()

    all_chunks = []

    # 1. Meta chunk
    # Title + meta description as standalone chunk
    # This ensures basic "what is this page about " queries hit something
    if title or meta_desc:
        meta_text = ''
        if title:
            meta_text += f"Page Title: {title}\n"
        if meta_desc:
            meta_desc += f"Summary: {meta_desc}"
        if len(meta_desc) >= MIN_CHUNK_SIZE:
            all_chunks.append({
                'text' : meta_text.strip(),
                'url' : url,
                'title' : title,
                'domain' : domain,
                'depth' : depth,
                'section' : 'Page Summary',
                'chunk_index' : 0,
                'source_type' : 'meta',
                'doc_id' : doc_id
            })
    
    # 2. Body text Chunk
    if body:
        ## split body into sections by markdown headings
        


text = """
Your chunker.py is implementing a hybrid semantic + structural chunking strategy. It is significantly more advanced than the basic RecursiveCharacterTextSplitter used in many LangChain examples.

I'll break down what method it is using, how it works, and whether it is good for your RAG crawler.
"""
sentences =_split_sentences(text)
print(_build_chunks_from_sentences(sentences, chunk_size=20, overlap=10))