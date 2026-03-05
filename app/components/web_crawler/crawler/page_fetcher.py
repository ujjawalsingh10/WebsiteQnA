"""
page_fetcher.py
------------------

Fetches pages via
- requests (for static HTML)

Features:
- Retry with backoff
- content-type detection
- redirects tracking
- response timing
- proper headers
"""

import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)

class FetchResult:
    """
    Structured result from fetch page
    """
    def __init__(self):
        self.url : str = ""
        self.final_url: str = "" # after redirects
        self.status_code: int = 0
        self.content_type: str = ""
        self.html: Optional[str] = None
        self.raw_bytes: Optional[bytes] = None
        self.is_html: bool = False
        self.is_pdf: bool = False
        self.is_image: bool = False
        self.redirected: bool = False
        self.redirect_chain: list = []
        self.response_time_ms: int = 0
        self.error: Optional[str] = None
        self.headers: dict = {}

    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.error is None
    
    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "is_html": self.is_html,
            "is_pdf": self.is_pdf,
            "is_image": self.is_image,
            "redirected": self.redirected,
            "redirect_chain": self.redirect_chain,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
        }

def _build_session(config: dict) -> requests.Session:
    """Build a requests seession with retry logic and proper headers
    Args:
        config (dict) : crawler configuration
    """
    crawl_cfg = config.get("crawl", {})
    max_retries = crawl_cfg.get("max_retries", 3)
    user_agent = crawl_cfg.get("user_agent", "RAGBot/1.0")

    session = requests.Session()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })

    return session

class PageFetcher:
    """
    Fetches URLs. Returns FetchResult
    Handles static HTML
    Args:
        config (dict): webcrawler configuration
    """

    def __init__(self, config: dict):
        self.config = config
        self.crawl_cfg = config.get("crawl", {})
        self.timeout = self.crawl_cfg.get("request_timeout_sec", 20)
        self.delay = self.crawl_cfg.get("delay_between_requests_sec", 1.5)

        self._session = _build_session(config)
        self._last_request_time: dict = {}  # domain → timestamp

    # ---------------
    # Main fetch
    # ----------------
    def fetch(self, url: str) -> FetchResult:
        """
        Fetch a URL
        Args:
            url (str): website url to fetch
        """
        self._rate_limit(url)
        domain = urlparse(url).hostname or ""
        return self._fetch_requests(url)
    

    # -----------------------
    # HTTP fetch
    # ------------------
    def _fetch_requests(self, url) -> FetchResult:
        result = FetchResult()
        result.url = url
        start = time.time()

        try:
            response = self._session.get(
                url, 
                timeout = self.timeout,
                allow_redirects=True,
                stream = False
            )

            result.response_time_ms = int((time.time() - start) * 1000)
            result.status_code = response.status_code
            result.final_url = response.url
            result.headers = dict(response.headers)
            result.redirected = len(response.history) > 0
            result.redirect_chain = [r.url for r in response.history]

            content_type = response.headers.get('Content-Type', '').lower()
            result.content_type = content_type

            result.is_html = 'text/html' in content_type or 'application/xhtml' in content_type
            result.is_pdf = 'application/pdf' in content_type
            result.is_image = any(t in content_type for t in [
                'image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'
            ])

            if response.status_code == 200:
                if result.is_html:
                    # detect encoding
                    result.html = response.text
                elif result.is_pdf or result.is_image:
                    result.raw_bytes = response.content
                else:
                    # unknown type - try as text
                    try:
                        result.html = response.text
                        result.is_html = True
                    except Exception:
                        result.raw_bytes = response.content
            else:
                result.error = f"HTTP {response.status_code}"
                logger.warning(f"HTTP {response.status_code} for {url}")
        except requests.exceptions.Timeout:
            result.error = "Timeout"
            result.response_time_ms = int((time.time() - start) * 1000)
            logger.warning(f"Timeout fetching {url}")

        except requests.exceptions.TooManyRedirects:
            result.error = "TooManyRedirects"
            logger.warning(f"Too many redirects: {url}")

        except requests.exceptions.ConnectionError as e:
            result.error = f"ConnectionError: {str(e)[:100]}"
            logger.warning(f"Connection error for {url}: {e}")

        except Exception as e:
            result.error = f"UnexpectedError: {str(e)[:100]}"
            logger.error(f"Unexpected error fetching {url}: {e}")

        return result


    # --------------------
    # Rate limiting
    # ----------------------

    def _rate_limit(self, url: str):
        """
        Per domain rate limiting.
        Ensures at least 'delay' seconds between requests to same domain
        """
        domain = urlparse(url).hostname or 'unknown'
        last = self._last_request_time.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s for {domain}")
            time.sleep(sleep_time)
        self._last_request_time[domain] = time.time()

    def head(self, url: str) -> dict:
        """
        Quick HEAD request to check content-type and size before downloading
        """
        try:
            resp = self._session.head(url, timeout=10, allow_redirects = True)
            return {
                "status_code": resp.status_code,
                "content_type": resp.headers.get("Content-Type", ""),
                "content_length": int(resp.headers.get("Content-Length", 0)),
                "final_url": resp.url,
            }
        except Exception as e:
            return {"status_code": 0, "error": str(e)}
        
    def close(self):
        """clean up resources"""
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()