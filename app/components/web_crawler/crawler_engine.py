"""
orchestrator.py
---------------
Main crawl orchestrator.
Ties together: URLFrontier → PageFetcher → PageParser → AssetDownloader → MetadataWriter

Flow per URL:
1. Pop from frontier
2. Fetch page
3. Parse HTML (text, links, images, PDFs)
4. Download images + PDFs
5. Write page JSON + update indexes
6. Add discovered links back to frontier
7. Repeat until queue empty or budget exhausted
"""

import logging
import time
from pathlib import Path
from urllib.parse import urlparse

from crawler.url_frontier import URLFrontier
from crawler.page_fetcher import PageFetcher
from crawler.page_parser import PageParser
from crawler.asset_downloader import AssetDownloader
from storage.metadata_writer import MetadataWriter

logger = logging.getLogger(__name__)


class CrawlOrchestrator:
    """
    Main controller for the web crawl.
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("output", {}).get("base_dir", "./output")
        self.log_every = config.get("logging", {}).get("progress_every_n_pages", 10)

        # Init components
        self.frontier = URLFrontier(config)
        self.fetcher = PageFetcher(config)
        self.parser = PageParser(config)
        self.writer = MetadataWriter(config, self.output_dir)
        self.downloader = AssetDownloader(
            config,
            session=self.fetcher._session,
            output_dir=self.output_dir
        )

        self._crawl_images = config.get("assets", {}).get("download_images", True)
        self._crawl_pdfs = config.get("assets", {}).get("download_pdfs", True)

    # ──────────────────────────────────────────────────────────────
    # Main run loop
    # ──────────────────────────────────────────────────────────────

    def run(self, seed_urls: list[str]):
        """Start crawl from seed URLs."""
        logger.info(f"Starting crawl — seeds: {seed_urls}")
        logger.info(f"Depth: {self.config['crawl']['max_depth']} | "
                    f"Max pages: {self.config['crawl']['max_pages']}")

        self.frontier.add_seeds(seed_urls)
        pages_processed = 0
        start_time = time.time()

        try:
            while self.frontier.has_items() and self.frontier.is_within_budget():
                item = self.frontier.pop()
                if item is None:
                    break

                url = item["url"]
                depth = item["depth"]

                # Skip if already crawled (resume mode)
                if self.writer.is_already_crawled(url):
                    logger.debug(f"Skipping (already crawled): {url}")
                    continue

                logger.debug(f"[depth={depth}] Crawling: {url}")

                # ── STEP 1: Fetch
                fetch_result = self.fetcher.fetch(url)

                # ── STEP 2: Handle non-HTML (PDF/image directly linked)
                if fetch_result.success and fetch_result.is_pdf:
                    self._handle_direct_pdf(item, fetch_result)
                    pages_processed += 1
                    continue

                if not fetch_result.success or not fetch_result.is_html:
                    self.writer.write_error(item, fetch_result)
                    logger.warning(f"Failed [{fetch_result.status_code}] [{fetch_result.error}]: {url}")
                    pages_processed += 1
                    continue

                # ── STEP 3: Parse HTML
                parsed = self.parser.parse(
                    html=fetch_result.html,
                    url=url,
                    final_url=fetch_result.final_url,
                )

                # ── STEP 4: Add discovered links to frontier
                all_links = parsed.internal_links + parsed.external_links
                self.frontier.add_links(
                    links=[{"url": l["url"], "text": l["text"]} for l in all_links],
                    from_url=url,
                    current_depth=depth,
                )

                # ── STEP 5: Download images
                image_assets = []
                if self._crawl_images:
                    domain = urlparse(url).hostname or ""
                    for img in parsed.images:
                        img_url = img.get("url", "")
                        if not img_url:
                            continue
                        dl_result = self.downloader.download_image(img_url, domain)
                        image_assets.append({**img, "download_result": dl_result.to_dict()})
                        if dl_result.success and not dl_result.skipped:
                            logger.debug(f"  ↳ Image: {img_url} → {dl_result.file_path}")

                # ── STEP 6: Download PDFs
                pdf_assets = []
                if self._crawl_pdfs:
                    domain = urlparse(url).hostname or ""
                    for pdf in parsed.pdfs:
                        pdf_url = pdf.get("url", "")
                        if not pdf_url:
                            continue
                        dl_result = self.downloader.download_pdf(pdf_url, domain)
                        pdf_assets.append({**pdf, "download_result": dl_result.to_dict()})
                        if dl_result.success and not dl_result.skipped:
                            logger.debug(f"  ↳ PDF: {pdf_url} → {dl_result.file_path}")

                # ── STEP 7: Write metadata
                file_path = self.writer.write_page(
                    queue_item=item,
                    fetch_result=fetch_result,
                    parsed_page=parsed,
                    image_assets=image_assets,
                    pdf_assets=pdf_assets,
                )

                pages_processed += 1

                # Progress logging
                if pages_processed % self.log_every == 0:
                    elapsed = time.time() - start_time
                    stats = self.frontier.stats()
                    logger.info(
                        f"Progress: {pages_processed} pages | "
                        f"Queue: {stats['queued']} | "
                        f"Elapsed: {elapsed:.0f}s | "
                        f"Rate: {pages_processed/elapsed:.1f} pg/s"
                    )
                    # Flush indexes periodically
                    self.writer.flush_indexes()

        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")

        finally:
            # Final flush
            self.writer.flush_indexes()
            stats = self.frontier.stats()
            summary = self.writer.write_crawl_summary(stats)

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("CRAWL COMPLETE")
            logger.info(f"  Pages crawled:    {pages_processed}")
            logger.info(f"  URLs seen:        {stats['visited']}")
            logger.info(f"  Errors:           {summary.get('errors', 0)}")
            logger.info(f"  Images saved:     {summary.get('total_images_downloaded', 0)}")
            logger.info(f"  PDFs saved:       {summary.get('total_pdfs_downloaded', 0)}")
            logger.info(f"  Time elapsed:     {elapsed:.1f}s")
            logger.info(f"  Output dir:       {self.output_dir}")
            logger.info("=" * 60)

            self.fetcher.close()

        return summary

    def _handle_direct_pdf(self, item: dict, fetch_result):
        """Handle a URL that directly returned a PDF (not linked from HTML)."""
        domain = urlparse(item["url"]).hostname or ""
        # Use raw bytes from fetch
        if fetch_result.raw_bytes:
            # Write bytes and register
            import hashlib
            from pathlib import Path
            content_hash = hashlib.sha256(fetch_result.raw_bytes).hexdigest()[:16]
            pdfs_dir = Path(self.output_dir) / "pdfs" / domain
            pdfs_dir.mkdir(parents=True, exist_ok=True)
            file_path = pdfs_dir / f"{content_hash}.pdf"
            file_path.write_bytes(fetch_result.raw_bytes)
            logger.info(f"Direct PDF: {item['url']} → {file_path}")