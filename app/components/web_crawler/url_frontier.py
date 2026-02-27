"""
url_frontier.py
---------------
BFS-based URL queue with:
- Depth tracking per URL
- Visited set (deduplication by normalized URL)
- Domain scope enforcement
- robots.txt compliance
- URL normalization and filtering
"""

import re
import hashlib
import logging
from typing import Optional
from collections import deque
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode

logger = logging.getLogger(__name__)

def normalize_url(url: str) -> str:
    """
    Normalize a URL for duplication
    - lowercase scheme and hostname (scheme=https, http | hostname = pmjay.gov.in)
    - remove default ports (80, 443) to remove duplication
    - remove fragments (#section)
    - sort query params
    - Remove trailing slash from path
    Args:
        url (str) : The website URL
    Returns:
        str: A normalized URL
    """

    try:
        logger.info(f"Normalizing URL: {url} !")

        parsed = urlparse(url.strip())

        # lower scheme (https, http) and hostname
        scheme = parsed.scheme.lower()
        host = parsed.hostname.lower() if parsed.hostname else ""

        # remove default ports
        port = parsed.port
        if (scheme == 'http' and port == 80) or (scheme == 'https' and port == 443):
            port = None
        netloc = f"{host}:{port}" if port else host

        # normalize path
        path  = parsed.path or '/'
        if path != '/' and path.endswith('/'):
            path.rstrip('/')
        
        # sort query params for consistency
        # example.com/page?b=2&a=1 | example.com/page?a=1&b=2
        query =''
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            sorted_params = sorted(params.items())
            query = urlencode([(k,v) for k,vals in sorted_params for v in vals])

        # drop fragments entirely
        # page#about | page#contact
        normalized = urlunparse((scheme, netloc, path, '', query, ''))
        return normalized
    
    except Exception as e:
        logging.exception('URL  normalization failed')
        return url

def url_hash(url: str) -> str:
    """
    Short hash of normalized URL for use as filename
    Args:
        url (str): website URL
    Returns:
        str: hash for filename
    """
    return hashlib.md5(normalize_url(url).encode()).hexdigest()[:12]

class URLFrontier:
    """
    BFS frontier
    Queue items: (url, depth, parent url, anchor_text)
    """

    def __init__(self, config: dict):
        self.config = config
        self.crawl_cfg = config.get('crawl', {})
        self.scope_cfg = config.get("scope", {})

        self.max_depth = self.crawl_cfg.get("max_depth", 3)
        self.max_pages = self.crawl_cfg.get("max_pages", 500)
        self.max_per_domain = self.crawl_cfg.get("max_pages_per_domain", 200)
        self.internal_only = self.scope_cfg.get("internal_only", True)
        self.allow_subdomains = self.scope_cfg.get("allow_subdomains", True)

        self.queue: deque = deque()

        # visited normalized urls
        self.visited: set = set()

        # count per domain
        self.domain_counts : dict = {}

        # seed domains (set from seed URLs)
        self.seed_domains: set = set()

        # external whitelist
        self.external_whitelist: set = set(
            self.scope_cfg.get('external_sites_whitelist', [])
        )

        # compiled exclude patterns = [
        self.exclude_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.scope_cfg.get('url_patterns_exclude', [])
        ]

        self.include_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.scope_cfg.get('url_patterns_include', [])
        ] if self.scope_cfg.get('url_patterns_include') else []

        ## Robots parsers per domain
        # self._robots_cache : dict = {}
        # self.respect_robots = self.crawl_cfg.get('respect_robots_txt', True)
        self.user_agent = self.crawl_cfg.get('user_agent', 'RAGBot/1.0')

        #stats
        self.total_enqueued = 0
        self.total_skipped = 0
    
    def add_seeds(self, urls: list[str]):
        """
        Add initial seed URLs and register their domains as 'home' domains
        Args:
            urls (list[str]) : list of urls
        """
        for url in urls:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            self.seed_domains.add(self._root_domain(host))
            self._enqueue(url, depth = 0, parent_url = None, anchor_text = "[SEED]")







    #------------------------
    # Internal Helpers
    #--------------------------

    
    def _enqueue(self, url: str, depth: int, parent_url: Optional[str], anchor_text: str):
        """
        Validate and add URL to queue
        Args:

        """
        try:
            #basic cleanup
            url = url.strip()
            if not url or url.startswith(('javascript:', "mailto:", "tel:", "data:", "#")):
                return
            
            # resolve relative URLs
            if parent_url and not url.startswith(('https://', "https://")):
                url = urljoin(parent_url, url)
            
            ## must be http/https
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return 
            
            # normalize for deduplication check
            norm = normalize_url(url)
            if norm in self.visited:
                return
            
            # scope check
            if not self._is_in_scope(url, parsed):
                self.total_skipped += 1
                return
            
            # pattern exclude check
            for pattern in self.exclude_patterns:
                if pattern.search(url):
                    logger.debug(f"Excluded by pattern: {url}")
                    self.total_skipped += 1
                    return 
            
            # pattern include check (if specified, URL must match at least one)
            if self.include_patterns:
                if not any(p.search(url) for p in self.include_patterns):
                    self.total_skipped += 1
                    return 
            
            ## robots.txt check
            # if self.respect_robots and not self._robots_allowed(url, parsed):
            #         logger.debug(f"Blocked by robots.txt: {url}")
            #         self.total_skipped += 1
            #         return

            # domain budget
            domain = parsed.hostname or ""
            if self.domain_counts.get(domain, 0) >= self.max_per_domain:
                self.total_skipped += 1
                return
            
            # add to queue
            self.queue.append({
                'url': url,
                'normalized_url': norm,
                'depth': depth,
                "parent_url": parent_url,
                'anchor_text' : anchor_text,
                'url_hash' : url_hash(url)
            })
            
            self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
            self.total_enqueued += 1
        
        except Exception as e:
            logger.warning(f"Error enqueuing {url}: {e}")

    def _root_domain(self, hostname: str) -> str:
        """
        Extract root domain e.g 'sub.example.gov.in' -> example.gov.in

        """
        parts = hostname.lower().split('.')
        if len(parts) >= 3 and parts[-2] in ('gov', 'co', 'org', 'net', 'edu', 'ac'):
            return ".".join(parts[-3:])
        return ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    

    def _is_in_scope(self, url: str, parsed) -> bool:
        """
        Check if URL is withing allowed crawl scope
        """
        host = parsed.hostname or ""
        root = self._root_domain(host)

        ## check if it is an internal domain ?
        is_internal = root in self.seed_domains or (
            self.allow_subdomains and any(
                host.endswith('.' + sd) or host == sd
                for sd in self.seed_domains
            )
        )

        if is_internal:
            return True
        
        ## external - check wishlist
        if not self.internal_only and root in self.external_whitelist:
            return True
        
        return False
    
    # def _robots_allowed(self, url: str, parsed) -> bool:
    #     """Check robots.txt for the given URL."""
    #     try:
    #         base = f"{parsed.scheme}://{parsed.netloc}"
    #         if base not in self._robots_cache:
    #             rp = urllib.robotparser.RobotFileParser()
    #             rp.set_url(f"{base}/robots.txt")
    #             try:
    #                 rp.read()
    #             except Exception:
    #                 rp = None  # If can't fetch robots.txt, allow
    #             self._robots_cache[base] = rp

    #         rp = self._robots_cache.get(base)
    #         if rp is None:
    #             return True
    #         return rp.can_fetch(self.user_agent, url)
    #     except Exception:
    #         return True  # Fail open