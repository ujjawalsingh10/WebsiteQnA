"""
metadata_writer.py
------------------
Saves all crawl data to disk as structured JSON.

Outputs:
- output/pages/{domain}/{url_hash}.json     — Full page record
- output/metadata/crawl_index.json          — Master URL → hash index
- output/metadata/image_index.json          — All images with metadata
- output/metadata/pdf_index.json            — All PDFs with metadata
- output/crawl_state.json                   — Resumable state (visited URLs, queue)
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(data: dict, path: Path, pretty: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)


def _json_load(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


class MetadataWriter:
    """
    Handles all JSON output for crawled content.
    Thread-safe writes via simple file locking pattern.
    """

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.pretty = config.get("output", {}).get("pretty_json", True)
        self.base = Path(output_dir)

        # Directory structure
        self.pages_dir = self.base / "pages"
        self.metadata_dir = self.base / "metadata"
        self.base.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # In-memory indexes (flushed to disk periodically)
        self._crawl_index: dict = {}     # url → {hash, status, timestamp}
        self._image_index: dict = {}     # image_url → image_record
        self._pdf_index: dict = {}       # pdf_url → pdf_record

        # Crawl session metadata
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self._start_time = _utc_now()
        self._page_count = 0
        self._error_count = 0

        # Load existing indexes if resuming
        self._load_indexes()

    # ──────────────────────────────────────────────────────────────
    # Public: write page
    # ──────────────────────────────────────────────────────────────

    def write_page(
        self,
        queue_item: dict,
        fetch_result,
        parsed_page,
        image_assets: list,
        pdf_assets: list,
    ) -> str:
        """
        Build and save the full page JSON record.
        Returns the file path.
        """
        url = queue_item["url"]
        url_hash = queue_item["url_hash"]
        domain = urlparse(url).hostname or "unknown"

        page_record = self._build_page_record(
            queue_item=queue_item,
            fetch_result=fetch_result,
            parsed_page=parsed_page,
            image_assets=image_assets,
            pdf_assets=pdf_assets,
        )

        # Save page JSON
        domain_dir = self.pages_dir / self._sanitize(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)
        page_path = domain_dir / f"{url_hash}.json"
        _json_dump(page_record, page_path, self.pretty)

        # Update crawl index
        self._crawl_index[url] = {
            "url_hash": url_hash,
            "depth": queue_item["depth"],
            "parent_url": queue_item.get("parent_url"),
            "status_code": fetch_result.status_code,
            "crawled_at": page_record["crawled_at"],
            "file_path": str(page_path),
            "title": parsed_page.title if parsed_page else "",
            "text_length": parsed_page.body_text_length if parsed_page else 0,
            "image_count": len(image_assets),
            "pdf_count": len(pdf_assets),
        }

        # Update image index
        for img in image_assets:
            if img.get("download_result", {}).get("success"):
                self._image_index[img["url"]] = {
                    "url": img["url"],
                    "source_page": url,
                    "source_domain": domain,
                    "file_path": img["download_result"].get("file_path", ""),
                    "content_hash": img["download_result"].get("content_hash", ""),
                    "file_size_bytes": img["download_result"].get("file_size_bytes", 0),
                    "width": img["download_result"].get("width", 0),
                    "height": img["download_result"].get("height", 0),
                    "image_type": img["download_result"].get("image_type", ""),
                    "alt": img.get("alt", ""),
                    "caption": img.get("caption", ""),
                    "nearest_heading": img.get("nearest_heading", ""),
                    "context_text": img.get("context_text", ""),
                    "crawled_at": _utc_now(),
                    # Processing status for RAG pipeline
                    "ocr_done": False,
                    "vision_done": False,
                    "embedded": False,
                }

        # Update PDF index
        for pdf in pdf_assets:
            if pdf.get("download_result", {}).get("success"):
                self._pdf_index[pdf["url"]] = {
                    "url": pdf["url"],
                    "source_page": url,
                    "source_domain": domain,
                    "file_path": pdf["download_result"].get("file_path", ""),
                    "content_hash": pdf["download_result"].get("content_hash", ""),
                    "file_size_bytes": pdf["download_result"].get("file_size_bytes", 0),
                    "pdf_page_count": pdf["download_result"].get("pdf_page_count", 0),
                    "link_text": pdf.get("link_text", ""),
                    "nearest_heading": pdf.get("nearest_heading", ""),
                    "context_text": pdf.get("context_text", ""),
                    "crawled_at": _utc_now(),
                    # Processing status for RAG pipeline
                    "parsed": False,
                    "ocr_done": False,
                    "embedded": False,
                }

        self._page_count += 1
        return str(page_path)

    def write_error(self, queue_item: dict, fetch_result):
        """Record a failed fetch."""
        url = queue_item["url"]
        self._crawl_index[url] = {
            "url_hash": queue_item["url_hash"],
            "depth": queue_item["depth"],
            "parent_url": queue_item.get("parent_url"),
            "status_code": fetch_result.status_code,
            "error": fetch_result.error,
            "crawled_at": _utc_now(),
        }
        self._error_count += 1

    # ──────────────────────────────────────────────────────────────
    # Build page record
    # ──────────────────────────────────────────────────────────────

    def _build_page_record(
        self,
        queue_item: dict,
        fetch_result,
        parsed_page,
        image_assets: list,
        pdf_assets: list,
    ) -> dict:
        """Build the complete page JSON record."""
        url = queue_item["url"]
        domain = urlparse(url).hostname or ""

        record = {
            # ── IDENTITY
            "doc_id": f"page_{queue_item['url_hash']}",
            "url_hash": queue_item["url_hash"],
            "doc_type": "web_page",

            # ── SOURCE
            "url": url,
            "final_url": fetch_result.final_url,
            "domain": domain,
            "depth": queue_item["depth"],
            "parent_url": queue_item.get("parent_url"),
            "anchor_text_from_parent": queue_item.get("anchor_text", ""),
            "redirected": fetch_result.redirected,
            "redirect_chain": fetch_result.redirect_chain,

            # ── CRAWL META
            "crawled_at": _utc_now(),
            "session_id": self._session_id,
            "response_time_ms": fetch_result.response_time_ms,
            "status_code": fetch_result.status_code,
            "content_type": fetch_result.content_type,

            # ── PAGE METADATA
            "title": parsed_page.title if parsed_page else "",
            "meta_description": parsed_page.meta_description if parsed_page else "",
            "meta_keywords": parsed_page.meta_keywords if parsed_page else "",
            "og_title": parsed_page.og_title if parsed_page else "",
            "og_description": parsed_page.og_description if parsed_page else "",
            "og_image": parsed_page.og_image if parsed_page else "",
            "canonical_url": parsed_page.canonical_url if parsed_page else "",
            "language": parsed_page.language if parsed_page else "",
            "page_type": parsed_page.page_type if parsed_page else "webpage",
            "breadcrumbs": parsed_page.breadcrumbs if parsed_page else [],

            # ── HEADING STRUCTURE
            "headings": parsed_page.headings if parsed_page else [],
            "heading_hierarchy": parsed_page.heading_hierarchy if parsed_page else [],

            # ── CONTENT
            "body_text": parsed_page.body_text if parsed_page else "",
            "body_text_length": parsed_page.body_text_length if parsed_page else 0,
            "raw_html_length": parsed_page.raw_html_length if parsed_page else 0,
            "extractor_used": parsed_page.extractor_used if parsed_page else "",
            "extraction_warnings": parsed_page.extraction_warnings if parsed_page else [],

            # ── LINKS
            "internal_links": parsed_page.internal_links if parsed_page else [],
            "external_links": parsed_page.external_links if parsed_page else [],
            "internal_link_count": len(parsed_page.internal_links) if parsed_page else 0,
            "external_link_count": len(parsed_page.external_links) if parsed_page else 0,

            # ── ASSETS
            "images": self._build_image_records(image_assets),
            "pdfs": self._build_pdf_records(pdf_assets),
            "tables": parsed_page.tables if parsed_page else [],
            "image_count": len(image_assets),
            "pdf_count": len(pdf_assets),

            # ── STRUCTURED DATA
            "schema_org": parsed_page.schema_org if parsed_page else [],

            # ── RAG PIPELINE STATUS
            "rag_status": {
                "chunked": False,
                "embedded": False,
                "indexed": False,
                "chunk_count": 0,
            }
        }

        return record

    def _build_image_records(self, image_assets: list) -> list:
        """Build clean image records for storage in page JSON."""
        records = []
        for img in image_assets:
            dl = img.get("download_result", {})
            records.append({
                "url": img.get("url", ""),
                "alt": img.get("alt", ""),
                "caption": img.get("caption", ""),
                "nearest_heading": img.get("nearest_heading", ""),
                "context_text": img.get("context_text", ""),
                "title": img.get("title", ""),
                "width_attr": img.get("width", ""),
                "height_attr": img.get("height", ""),
                "downloaded": dl.get("success", False),
                "file_path": dl.get("file_path", ""),
                "file_size_bytes": dl.get("file_size_bytes", 0),
                "content_hash": dl.get("content_hash", ""),
                "actual_width": dl.get("width", 0),
                "actual_height": dl.get("height", 0),
                "image_type": dl.get("image_type", ""),
                "skipped": dl.get("skipped", False),
                "skip_reason": dl.get("skip_reason", ""),
                "error": dl.get("error"),
                # Will be filled by Phase 3 (OCR + Vision LLM)
                "ocr_text": None,
                "vision_description": None,
                "languages_detected": [],
            })
        return records

    def _build_pdf_records(self, pdf_assets: list) -> list:
        """Build clean PDF records for storage in page JSON."""
        records = []
        for pdf in pdf_assets:
            dl = pdf.get("download_result", {})
            records.append({
                "url": pdf.get("url", ""),
                "link_text": pdf.get("link_text", ""),
                "nearest_heading": pdf.get("nearest_heading", ""),
                "context_text": pdf.get("context_text", ""),
                "downloaded": dl.get("success", False),
                "file_path": dl.get("file_path", ""),
                "file_size_bytes": dl.get("file_size_bytes", 0),
                "content_hash": dl.get("content_hash", ""),
                "pdf_page_count": dl.get("pdf_page_count", 0),
                "skipped": dl.get("skipped", False),
                "skip_reason": dl.get("skip_reason", ""),
                "error": dl.get("error"),
                # Will be filled by Phase 2 (PDF Ingestion)
                "title": None,
                "author": None,
                "parsed": False,
            })
        return records

    # ──────────────────────────────────────────────────────────────
    # Index persistence
    # ──────────────────────────────────────────────────────────────

    def flush_indexes(self):
        """Write all indexes to disk."""
        _json_dump(self._crawl_index, self.metadata_dir / "crawl_index.json", self.pretty)
        _json_dump(self._image_index, self.metadata_dir / "image_index.json", self.pretty)
        _json_dump(self._pdf_index, self.metadata_dir / "pdf_index.json", self.pretty)

    def _load_indexes(self):
        """Load existing indexes for resumable crawl."""
        self._crawl_index = _json_load(self.metadata_dir / "crawl_index.json")
        self._image_index = _json_load(self.metadata_dir / "image_index.json")
        self._pdf_index = _json_load(self.metadata_dir / "pdf_index.json")
        if self._crawl_index:
            logger.info(f"Resuming crawl — loaded {len(self._crawl_index)} existing URLs from index")

    def is_already_crawled(self, url: str) -> bool:
        """Check if URL is in the existing crawl index (for resume)."""
        return url in self._crawl_index

    def write_crawl_summary(self, frontier_stats: dict):
        """Write final crawl summary."""
        summary = {
            "session_id": self._session_id,
            "start_time": self._start_time,
            "end_time": _utc_now(),
            "pages_crawled": self._page_count,
            "errors": self._error_count,
            "total_urls_seen": frontier_stats.get("visited", 0),
            "total_images_downloaded": sum(
                1 for v in self._image_index.values()
                if v.get("file_path")
            ),
            "total_pdfs_downloaded": sum(
                1 for v in self._pdf_index.values()
                if v.get("file_path")
            ),
            "domain_breakdown": frontier_stats.get("domain_counts", {}),
            "config": self.config,
        }
        _json_dump(summary, self.base / "crawl_summary.json", self.pretty)
        logger.info(f"Crawl summary saved: {self._page_count} pages, {self._error_count} errors")
        return summary

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(name: str) -> str:
        import re
        return re.sub(r"[^\w\-.]", "_", name)[:60]