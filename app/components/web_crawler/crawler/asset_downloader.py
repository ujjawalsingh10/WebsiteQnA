"""
asset_downloader.py
-------------------
Downloads images and PDFs found during crawling.
- Validates content type before saving
- Generates content hash (dedup)
- Extracts image dimensions
- Classifies image type (poster, photo, icon, etc.)
- Extracts basic PDF metadata
- Size filtering
"""

import os
import io
import hashlib
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


IMAGE_MIME_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

PDF_MIME = "application/pdf"


class DownloadedAsset:
    def __init__(self):
        self.url: str = ""
        self.content_hash: str = ""
        self.file_path: str = ""
        self.file_size_bytes: int = 0
        self.content_type: str = ""
        self.extension: str = ""
        self.success: bool = False
        self.skipped: bool = False
        self.skip_reason: str = ""
        self.error: Optional[str] = None
        # Image-specific
        self.width: int = 0
        self.height: int = 0
        self.image_type: str = ""     # poster | photo | icon | infographic | diagram | unknown
        # PDF-specific
        self.pdf_page_count: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class AssetDownloader:
    """
    Downloads images and PDFs.
    Uses the PageFetcher's session for actual HTTP requests.
    """

    def __init__(self, config: dict, session, output_dir: str):
        """
        config: full crawl config
        session: requests.Session (from PageFetcher)
        output_dir: base output directory
        """
        self.config = config
        self.assets_cfg = config.get("assets", {})
        self.session = session

        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.pdfs_dir = self.output_dir / "pdfs"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

        self.min_image_bytes = self.assets_cfg.get("min_image_size_bytes", 5120)   # 5KB
        self.max_image_mb = self.assets_cfg.get("max_image_size_mb", 20)
        self.max_pdf_mb = self.assets_cfg.get("max_pdf_size_mb", 100)

        # Track downloaded hashes to avoid re-saving duplicates
        self._downloaded_hashes: dict = {}   # hash → file_path

    # ──────────────────────────────────────────────────────────────
    # Image download
    # ──────────────────────────────────────────────────────────────

    def download_image(self, url: str, source_domain: str) -> DownloadedAsset:
        result = DownloadedAsset()
        result.url = url

        try:
            # HEAD check first to avoid downloading trash
            head = self._head(url)
            if not head:
                result.error = "HEAD request failed"
                return result

            content_type = head.get("content_type", "").split(";")[0].strip().lower()
            content_length = head.get("content_length", 0)

            # Size check
            if content_length > 0:
                if content_length < self.min_image_bytes:
                    result.skipped = True
                    result.skip_reason = f"Too small ({content_length}B < {self.min_image_bytes}B)"
                    return result
                if content_length > self.max_image_mb * 1024 * 1024:
                    result.skipped = True
                    result.skip_reason = f"Too large (>{self.max_image_mb}MB)"
                    return result

            # Validate MIME type
            if content_type and not content_type.startswith("image/"):
                # Try to infer from URL
                ext = Path(urlparse(url).path).suffix.lower()
                if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                    result.skipped = True
                    result.skip_reason = f"Not an image: {content_type}"
                    return result

            # Download
            response = self.session.get(url, timeout=30, stream=True)
            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}"
                return result

            data = response.content

            # Content-type from actual response
            actual_ct = response.headers.get("Content-Type", content_type).split(";")[0].strip().lower()
            result.content_type = actual_ct

            # Size validation after download
            if len(data) < self.min_image_bytes:
                result.skipped = True
                result.skip_reason = f"Too small after download ({len(data)}B)"
                return result

            # Hash for dedup
            content_hash = hashlib.sha256(data).hexdigest()[:16]
            result.content_hash = content_hash

            if content_hash in self._downloaded_hashes:
                result.file_path = self._downloaded_hashes[content_hash]
                result.success = True
                result.skipped = True
                result.skip_reason = "Duplicate (hash match)"
                return result

            # Extension
            ext = IMAGE_MIME_MAP.get(actual_ct, "")
            if not ext:
                ext = Path(urlparse(url).path).suffix.lower()
            if not ext:
                ext = ".jpg"
            result.extension = ext

            # Save
            domain_dir = self.images_dir / self._sanitize_domain(source_domain)
            domain_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{content_hash}{ext}"
            file_path = domain_dir / file_name

            file_path.write_bytes(data)
            result.file_path = str(file_path)
            result.file_size_bytes = len(data)
            self._downloaded_hashes[content_hash] = str(file_path)

            # Image analysis
            try:
                img = PILImage.open(io.BytesIO(data))
                result.width, result.height = img.size
                result.image_type = self._classify_image(img, result.width, result.height, len(data))
            except Exception:
                pass

            if not result.image_type:
                result.image_type = self._classify_image_heuristic(
                    url, result.width, result.height, len(data)
                )

            result.success = True

        except Exception as e:
            result.error = str(e)[:200]
            logger.warning(f"Failed to download image {url}: {e}")

        return result

    def _classify_image(self, img, width: int, height: int, size_bytes: int) -> str:
        """Classify image type using PIL."""
        # Icon: small
        if width <= 64 or height <= 64:
            return "icon"
        if width <= 200 and height <= 200:
            return "icon"

        # Likely a poster/banner: very wide
        aspect = width / max(height, 1)
        if aspect > 3:
            return "banner"
        if aspect < 0.4:
            return "poster_vertical"

        # Large image
        if width >= 800 and height >= 600:
            return "photo_or_poster"

        return "image_unknown"

    def _classify_image_heuristic(self, url: str, width: int, height: int, size_bytes: int) -> str:
        """Classify image type from URL patterns and size hints."""
        url_lower = url.lower()
        if any(k in url_lower for k in ["icon", "logo", "favicon", "sprite", "thumb"]):
            return "icon_or_logo"
        if any(k in url_lower for k in ["banner", "hero", "header", "cover"]):
            return "banner"
        if any(k in url_lower for k in ["poster", "notice", "circular", "flyer", "scheme"]):
            return "poster"
        if any(k in url_lower for k in ["photo", "image", "img", "pic", "gallery"]):
            return "photo"
        if any(k in url_lower for k in ["chart", "graph", "infographic", "diagram"]):
            return "infographic_or_chart"
        return "image_unknown"

    # ──────────────────────────────────────────────────────────────
    # PDF download
    # ──────────────────────────────────────────────────────────────

    def download_pdf(self, url: str, source_domain: str) -> DownloadedAsset:
        result = DownloadedAsset()
        result.url = url
        result.content_type = "application/pdf"
        result.extension = ".pdf"

        try:
            # HEAD check
            head = self._head(url)
            if head:
                content_length = head.get("content_length", 0)
                if content_length > self.max_pdf_mb * 1024 * 1024:
                    result.skipped = True
                    result.skip_reason = f"Too large (>{self.max_pdf_mb}MB)"
                    return result

            # Download
            response = self.session.get(url, timeout=60, stream=True)
            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}"
                return result

            # Verify it's actually a PDF
            data = response.content
            if not data.startswith(b"%PDF"):
                result.skipped = True
                result.skip_reason = "Not a valid PDF (missing %PDF header)"
                return result

            if len(data) > self.max_pdf_mb * 1024 * 1024:
                result.skipped = True
                result.skip_reason = f"Too large (>{self.max_pdf_mb}MB)"
                return result

            # Hash for dedup
            content_hash = hashlib.sha256(data).hexdigest()[:16]
            result.content_hash = content_hash

            if content_hash in self._downloaded_hashes:
                result.file_path = self._downloaded_hashes[content_hash]
                result.success = True
                result.skipped = True
                result.skip_reason = "Duplicate (hash match)"
                return result

            # Save
            domain_dir = self.pdfs_dir / self._sanitize_domain(source_domain)
            domain_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{content_hash}.pdf"
            file_path = domain_dir / file_name

            file_path.write_bytes(data)
            result.file_path = str(file_path)
            result.file_size_bytes = len(data)
            self._downloaded_hashes[content_hash] = str(file_path)

            # Basic PDF metadata (page count from xref)
            result.pdf_page_count = self._count_pdf_pages(data)

            result.success = True

        except Exception as e:
            result.error = str(e)[:200]
            logger.warning(f"Failed to download PDF {url}: {e}")

        return result

    def _count_pdf_pages(self, data: bytes) -> int:
        """Quick page count from raw PDF bytes without full parsing."""
        try:
            # Count /Page objects (approximation)
            count = data.count(b"/Type /Page\n") + data.count(b"/Type/Page\n") + data.count(b"/Type /Page\r")
            if count > 0:
                return count
            # Fallback: look for /Count in Pages dict
            import re
            matches = re.findall(rb"/Count\s+(\d+)", data)
            if matches:
                return int(matches[-1])
        except Exception:
            pass
        return 0

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _head(self, url: str) -> Optional[dict]:
        try:
            resp = self.session.head(url, timeout=10, allow_redirects=True)
            return {
                "status_code": resp.status_code,
                "content_type": resp.headers.get("Content-Type", ""),
                "content_length": int(resp.headers.get("Content-Length", 0) or 0),
                "final_url": resp.url,
            }
        except Exception:
            return None

    def _sanitize_domain(self, domain: str) -> str:
        """Make domain safe for use as directory name."""
        return re.sub(r"[^\w\-.]", "_", domain)[:50] if domain else "unknown"


import re  # Needed for sanitize_domain