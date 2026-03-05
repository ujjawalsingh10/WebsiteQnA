"""
main.py
-------
Entry point for the web crawler.

Usage:
    python main.py --url https://example.gov.in --depth 3
    python main.py --config config/crawl_config.json
    python main.py --url https://example.gov.in --dry-run
    python main.py --url https://example.gov.in --depth 2 --output ./my_output
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = None):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("charset_normalizer").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────
# Default config
# ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "seeds": [],
    "crawl": {
        "max_depth": 3,
        "max_pages": 500,
        "max_pages_per_domain": 200,
        "delay_between_requests_sec": 1.5,
        "request_timeout_sec": 20,
        "max_retries": 3,
        "concurrent_requests": 1,
        "respect_robots_txt": True,
        "user_agent": "RAGBot/1.0 (+https://yoursite.com/bot)",
    },
    "scope": {
        "internal_only": True,
        "allow_subdomains": True,
        "external_sites_whitelist": [],
        "url_patterns_include": [],
        "url_patterns_exclude": [
            r".*\.(css|js|woff|woff2|ttf|eot|ico|svg)$",
            r".*/login.*",
            r".*/logout.*",
            r".*/cart.*",
            r".*/account.*",
            r".*\?.*sessionid.*",
        ],
    },
    "assets": {
        "download_images": True,
        "download_pdfs": True,
        "min_image_size_bytes": 5120,
        "max_image_size_mb": 20,
        "max_pdf_size_mb": 100,
        "image_extensions": [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"],
        "pdf_extensions": [".pdf"],
    },
    "extraction": {
        "use_trafilatura": True,
        "preserve_heading_hierarchy": True,
        "extract_tables": True,
        "extract_meta_tags": True,
        "extract_structured_data": True,
        "min_text_length_chars": 100,
    },
    "js_rendering": {
        "enabled": False,
        "domains_requiring_js": [],
        "wait_for_selector": "body",
        "wait_timeout_ms": 5000,
    },
    "output": {
        "base_dir": "./output",
        "pretty_json": True,
    },
    "logging": {
        "level": "INFO",
        "log_file": "./output/crawl.log",
        "progress_every_n_pages": 10,
    },
}


# ──────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load config from file, merging with defaults."""
    config = dict(DEFAULT_CONFIG)
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            user_config = json.load(f)
        # Deep merge
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    return config


# ──────────────────────────────────────────────────────────────────
# Dry run
# ──────────────────────────────────────────────────────────────────

def dry_run(config: dict, seed_urls: list):
    """Discover and list URLs without downloading anything."""
    from crawler.url_frontier import URLFrontier
    from crawler.page_fetcher import PageFetcher
    from crawler.page_parser import PageParser

    logger = logging.getLogger("dry_run")
    logger.info("=== DRY RUN MODE — No files will be saved ===")

    # Disable asset downloads
    config["assets"]["download_images"] = False
    config["assets"]["download_pdfs"] = False
    config["crawl"]["max_pages"] = min(config["crawl"]["max_pages"], 50)

    frontier = URLFrontier(config)
    frontier.add_seeds(seed_urls)
    fetcher = PageFetcher(config)
    parser = PageParser(config)

    discovered_urls = []

    try:
        while frontier.has_items() and frontier.is_within_budget():
            item = frontier.pop()
            if not item:
                break
            url = item["url"]
            logger.info(f"[depth={item['depth']}] {url}")
            result = fetcher.fetch(url)
            if result.success and result.is_html:
                parsed = parser.parse(result.html, url)
                all_links = parsed.internal_links + parsed.external_links
                frontier.add_links(
                    [{"url": l["url"], "text": l["text"]} for l in all_links],
                    from_url=url,
                    current_depth=item["depth"],
                )
                discovered_urls.append(url)
    finally:
        fetcher.close()

    logger.info(f"\nDiscovered {len(discovered_urls)} URLs")
    logger.info(f"Frontier stats: {frontier.stats()}")
    return discovered_urls


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Web Crawler — Stage 1 of RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --url https://example.gov.in
  python main.py --url https://example.gov.in --depth 2 --max-pages 100
  python main.py --config config/crawl_config.json
  python main.py --url https://example.gov.in --dry-run
  python main.py --url https://example.gov.in --no-images --no-pdfs
        """
    )
    parser.add_argument("--url", type=str, help="Seed URL to start crawling")
    parser.add_argument("--urls", nargs="+", help="Multiple seed URLs")
    parser.add_argument("--config", type=str, default="config/crawl_config.json",
                        help="Path to config JSON (default: config/crawl_config.json)")
    parser.add_argument("--depth", type=int, help="Max crawl depth (overrides config)")
    parser.add_argument("--max-pages", type=int, help="Max pages to crawl (overrides config)")
    parser.add_argument("--output", type=str, help="Output directory (overrides config)")
    parser.add_argument("--delay", type=float, help="Delay between requests in seconds")
    parser.add_argument("--no-images", action="store_true", help="Skip image downloads")
    parser.add_argument("--no-pdfs", action="store_true", help="Skip PDF downloads")
    parser.add_argument("--allow-external", action="store_true",
                        help="Allow crawling external domains in whitelist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover URLs only, do not download or save")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.depth is not None:
        config["crawl"]["max_depth"] = args.depth
    if args.max_pages is not None:
        config["crawl"]["max_pages"] = args.max_pages
    if args.output:
        config["output"]["base_dir"] = args.output
    if args.delay is not None:
        config["crawl"]["delay_between_requests_sec"] = args.delay
    if args.no_images:
        config["assets"]["download_images"] = False
    if args.no_pdfs:
        config["assets"]["download_pdfs"] = False
    if args.allow_external:
        config["scope"]["internal_only"] = False

    # Determine seed URLs
    seed_urls = []
    if args.url:
        seed_urls.append(args.url)
    if args.urls:
        seed_urls.extend(args.urls)
    if not seed_urls and config.get("seeds"):
        seed_urls = config["seeds"]

    if not seed_urls:
        print("ERROR: No seed URLs provided. Use --url or set 'seeds' in config.")
        sys.exit(1)

    # Setup logging
    setup_logging(
        level=args.log_level or config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )

    logger = logging.getLogger("main")
    logger.info(f"Crawler starting | Seeds: {seed_urls}")

    # Run
    if args.dry_run:
        dry_run(config, seed_urls)
    else:
        from orchestrator import CrawlOrchestrator
        orchestrator = CrawlOrchestrator(config)
        orchestrator.run(seed_urls)


if __name__ == "__main__":
    main()