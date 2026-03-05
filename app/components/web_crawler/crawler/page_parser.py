"""
page_parser.py
--------------
Parses HTML pages and extracts:
- Clean Markdown body text (your approach: BS4 clean → markdownify)
- Headings extracted from original soup BEFORE cleaning
- Images + PDFs extracted from original soup BEFORE img tags are decomposed
- Internal/external links
- Page metadata (title, description, og tags, schema.org)
- Tables as Markdown

IMPORTANT ORDER OF OPERATIONS:
  1. Parse raw HTML → soup
  2. Extract metadata, headings, images, PDFs, links, tables  
  3. Then clean soup (decompose noise, remove imgs, etc.)      
  4. Convert cleaned soup → Markdown                           
  5. Post-process Markdown (strip blank lines, empty headings) 
"""

import re
import json
import copy
import logging
from urllib.parse import urljoin, urlparse
from html import unescape
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

from markdownify import markdownify as md
MARKDOWNIFY_AVAILABLE = True
# ─────────────────────────────────────────────────────────────────
# ParsedPage
# ─────────────────────────────────────────────────────────────────

class ParsedPage:
    """Structured result from parsing a page."""

    def __init__(self):
        # Identity
        self.url: str = ""
        self.final_url: str = ""

        # Page metadata
        self.title: str = ""
        self.meta_description: str = ""
        self.meta_keywords: str = ""
        self.og_title: str = ""
        self.og_description: str = ""
        self.og_image: str = ""
        self.canonical_url: str = ""
        self.language: str = ""
        self.page_type: str = "webpage"

        # Content
        self.raw_html_length: int = 0
        self.body_text: str = ""          # Final clean Markdown
        self.body_text_length: int = 0
        self.headings: list = []          # [{level, text, id}]
        self.heading_hierarchy: list = [] # ["H1 → H2 → H3"] paths
        self.breadcrumbs: list = []
        self.tables: list = []            # [{caption, markdown, headers, row_count}]

        # Links
        self.internal_links: list = []
        self.external_links: list = []

        # Assets — extracted BEFORE soup cleaning
        self.images: list = []
        self.pdfs: list = []

        # Structured data
        self.schema_org: list = []

        # Extraction meta
        self.extractor_used: str = "markdownify"
        self.extraction_warnings: list = []

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────
# PageParser
# ─────────────────────────────────────────────────────────────────

class PageParser:
    """
    Markdown-first HTML parser. Resilient to inconsistent HTML structure.

    Strategy:
      - All assets (images, PDFs, links, headings) are extracted
        from the ORIGINAL soup before any cleaning happens.
      - Then soup is cleaned and converted to Markdown for body text.
    """

    def __init__(self, config: dict):
        self.config = config
        self.extraction_cfg = config.get("extraction", {})
        self.assets_cfg = config.get("assets", {})

        self.min_text_length = self.extraction_cfg.get("min_text_length_chars", 100)
        self.image_extensions = set(self.assets_cfg.get("image_extensions", [
            ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"
        ]))

        # CSS selectors for site-specific noise blocks — add to config or here
        self.noise_selectors: list = self.extraction_cfg.get(
            "noise_selectors", [
                ".topbar",
                ".footer_logos",
            ]
        )

        # Anchor text prefixes that flag skip-to-content links
        self.skip_link_prefixes: tuple = tuple(
            self.extraction_cfg.get("skip_link_prefixes", ["skip", "jump to"])
        )

    # ─────────────────────────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────────────────────────

    def parse(self, html: str, url: str, final_url: str = "") -> ParsedPage:
        page = ParsedPage()
        page.url = url
        page.final_url = final_url or url
        page.raw_html_length = len(html)

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            page.extraction_warnings.append(f"HTML parse failed: {e}")
            return page

        # ── PHASE 1: Extract everything from ORIGINAL soup
        # Order matters — all of these must run BEFORE _clean_soup()
        self._extract_metadata(soup, page, url)
        self._extract_headings(soup, page)
        self._extract_images(soup, page, url)       # must be before img.decompose()
        self._extract_pdfs(soup, page, url)          # must be before link cleaning
        self._extract_links(soup, page, url)
        if self.extraction_cfg.get("extract_tables", True):
            self._extract_tables(soup, page)
        if self.extraction_cfg.get("extract_structured_data", True):
            self._extract_schema_org(soup, page)

        # ── PHASE 2: Clean → Markdown (on a copy)
        clean_soup = copy.copy(soup)
        self._clean_soup(clean_soup)
        page.body_text = self._to_markdown(clean_soup)
        page.body_text_length = len(page.body_text)

        if page.body_text_length < self.min_text_length:
            page.extraction_warnings.append(
                f"Short body text ({page.body_text_length} chars) — "
                "may be JS-rendered or mostly images"
            )

        return page

    # ─────────────────────────────────────────────────────────────
    # Phase 1 extractors
    # ─────────────────────────────────────────────────────────────

    def _extract_metadata(self, soup: BeautifulSoup, page: ParsedPage, url: str):
        """Title, meta tags, og tags, canonical, language."""
        title_tag = soup.find("title")
        page.title = title_tag.get_text(strip=True) if title_tag else ""

        html_tag = soup.find("html")
        if html_tag:
            page.language = html_tag.get("lang", "") or html_tag.get("xml:lang", "")

        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower().strip()
            content = (meta.get("content") or "").strip()
            if not content:
                continue
            if name == "description":
                page.meta_description = content
            elif name == "keywords":
                page.meta_keywords = content
            elif name == "og:title":
                page.og_title = content
            elif name == "og:description":
                page.og_description = content
            elif name == "og:image":
                page.og_image = content
            elif name == "og:type":
                page.page_type = content

        canonical = soup.find("link", rel="canonical")
        if canonical:
            page.canonical_url = canonical.get("href", "")

    def _extract_headings(self, soup: BeautifulSoup, page: ParsedPage):
        """
        Extract h1–h6 from ORIGINAL soup.
        Builds flat list + breadcrumb hierarchy for section-aware chunking.
        """
        headings = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text = tag.get_text(separator=" ", strip=True)
            if not text or len(text) < 2:
                continue
            headings.append({
                "level": int(tag.name[1]),
                "text": text[:200],
                "id": tag.get("id", ""),
            })
        page.headings = headings

        if headings:
            page.heading_hierarchy = self._build_heading_hierarchy(headings)

    def _build_heading_hierarchy(self, headings: list) -> list:
        """
        Build breadcrumb-style path for each heading.
        [{level:1,"Schemes"}, {level:2,"Housing"}, {level:3,"Eligibility"}]
        → ["Schemes", "Schemes → Housing", "Schemes → Housing → Eligibility"]
        """
        stack = []
        hierarchy = []
        for h in headings:
            level, text = h["level"], h["text"]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            hierarchy.append(" → ".join(t for _, t in stack))
        return hierarchy

    def _extract_images(self, soup: BeautifulSoup, page: ParsedPage, base_url: str):
        """
        Extract ALL <img> tags with full context from ORIGINAL soup.
        Must run before _clean_soup() which calls img.decompose().

        Captures:
          - src with lazy-load fallbacks (data-src, data-lazy-src, data-original)
          - alt, title
          - figcaption (if inside <figure>)
          - nearest heading above the image
          - surrounding paragraph/div text as context
        """
        seen_urls: set = set()

        # Build tag position map for "nearest heading" lookup
        all_tags = list(soup.find_all(True))
        tag_pos = {id(t): i for i, t in enumerate(all_tags)}

        headings_positioned = [
            (tag_pos.get(id(h), 0), h.get_text(strip=True))
            for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        ]

        for img in soup.find_all("img"):
            # src with common lazy-load attribute fallbacks
            src = (
                img.get("src") or
                img.get("data-src") or
                img.get("data-lazy-src") or
                img.get("data-original") or
                img.get("data-url") or
                ""
            ).strip()

            if not src or src.startswith("data:"):
                continue

            abs_url = urljoin(base_url, src)
            if abs_url in seen_urls:
                continue
            seen_urls.add(abs_url)

            # figcaption
            caption = ""
            figure = img.find_parent("figure")
            if figure:
                figcap = figure.find("figcaption")
                if figcap:
                    caption = figcap.get_text(strip=True)

            # Nearest heading above this image in document order
            img_pos = tag_pos.get(id(img), 0)
            nearest_heading = ""
            for h_pos, h_text in reversed(headings_positioned):
                if h_pos < img_pos:
                    nearest_heading = h_text
                    break

            # Context: walk up to nearest meaningful block parent
            context_text = ""
            for parent_tag_name in ["p", "div", "section", "article", "li", "td"]:
                parent = img.find_parent(parent_tag_name)
                if parent:
                    ctx = parent.get_text(separator=" ", strip=True)
                    if len(ctx) > 20:
                        context_text = ctx[:300]
                        break

            page.images.append({
                "url": abs_url,
                "alt": unescape(img.get("alt", "")).strip()[:200],
                "title": unescape(img.get("title", "")).strip()[:200],
                "width": img.get("width", ""),
                "height": img.get("height", ""),
                "loading": img.get("loading", ""),
                "caption": caption[:300],
                "nearest_heading": nearest_heading[:200],
                "context_text": context_text,
                "src_original": src,
            })

    def _extract_pdfs(self, soup: BeautifulSoup, page: ParsedPage, base_url: str):
        """
        Extract all PDF links with context from ORIGINAL soup.
        Must run before any link cleaning.
        """
        seen_urls: set = set()

        all_tags = list(soup.find_all(True))
        tag_pos = {id(t): i for i, t in enumerate(all_tags)}
        headings_positioned = [
            (tag_pos.get(id(h), 0), h.get_text(strip=True))
            for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        ]

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue

            abs_url = urljoin(base_url, href)
            path = urlparse(abs_url).path.lower()
            is_pdf = (
                path.endswith(".pdf") or
                "pdf" in a.get("type", "").lower() or
                "/pdf/" in path
            )
            if not is_pdf:
                continue

            if abs_url in seen_urls:
                continue
            seen_urls.add(abs_url)

            a_pos = tag_pos.get(id(a), 0)
            nearest_heading = ""
            for h_pos, h_text in reversed(headings_positioned):
                if h_pos < a_pos:
                    nearest_heading = h_text
                    break

            parent = a.find_parent(["p", "li", "div", "td"])
            context = parent.get_text(separator=" ", strip=True)[:200] if parent else ""

            page.pdfs.append({
                "url": abs_url,
                "link_text": a.get_text(strip=True)[:200],
                "nearest_heading": nearest_heading[:200],
                "context_text": context,
            })

    def _extract_links(self, soup: BeautifulSoup, page: ParsedPage, base_url: str):
        """
        Extract internal and external links.
        Skips: PDF links (captured above), skip-links, fragment/mailto anchors.
        """
        base_domain = urlparse(base_url).netloc or ""
        seen_urls: set = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
                continue

            abs_url = urljoin(base_url, href)
            parsed = urlparse(abs_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if abs_url in seen_urls:
                continue
            seen_urls.add(abs_url)

            # Skip PDF links — already captured
            if parsed.path.lower().endswith(".pdf"):
                continue

            # Skip skip-to-content links
            link_text = a.get_text(strip=True)
            if link_text.lower().startswith(self.skip_link_prefixes):
                continue

            rel = a.get("rel", [])
            rel_str = " ".join(rel) if isinstance(rel, list) else str(rel)
            entry = {"url": abs_url, "text": link_text[:200], "rel": rel_str}

            link_domain = parsed.netloc or ""
            if link_domain == base_domain or link_domain.endswith("." + base_domain):
                page.internal_links.append(entry)
            else:
                page.external_links.append(entry)

    def _extract_tables(self, soup: BeautifulSoup, page: ParsedPage):
        """
        Extract tables as Markdown from ORIGINAL soup.
        Tables are easier to parse here than from converted Markdown.
        """
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue

            caption_tag = table.find("caption")
            caption = caption_tag.get_text(strip=True) if caption_tag else ""

            # Headers — prefer <thead>, fall back to first row
            headers = []
            thead = table.find("thead")
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
            elif rows:
                headers = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]

            md_lines = []
            if headers:
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            data_rows = []
            for row in (rows[1:] if headers else rows):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if any(c.strip() for c in cells):
                    md_lines.append("| " + " | ".join(cells) + " |")
                    data_rows.append(cells)

            if not data_rows:
                continue

            page.tables.append({
                "caption": caption,
                "headers": headers,
                "row_count": len(data_rows),
                "markdown": "\n".join(md_lines),
            })

    def _extract_schema_org(self, soup: BeautifulSoup, page: ParsedPage):
        """Extract JSON-LD structured data blocks."""
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                page.schema_org.append(data)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────
    # Phase 2: Clean soup → Markdown
    # Your cleaning logic, properly sequenced
    # ─────────────────────────────────────────────────────────────

    def _clean_soup(self, soup: BeautifulSoup):
        """
        Remove all noise from soup IN PLACE before markdown conversion.

        Follows your approach exactly, with:
        - Proper ordering (broad tags first, specific selectors after)
        - img decompose moved here (safe now — already captured in Phase 1)
        - Hidden element removal
        - Comment stripping
        """

        # 1. Standard noise tags — broad removal first
        for tag in soup.find_all([
            "script", "style", "header", "footer",
            "noscript", "iframe", "embed", "object",
            "svg", "canvas",
            # Note: nav, aside, form kept commented like your original
            # uncomment if your site's nav bleeds into content:
            # "nav", "aside", "form",
        ]):
            tag.decompose()

        # 2. HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 3. Site-specific class/id selectors (your .topbar, .footer_logos, etc.)
        for selector in self.noise_selectors:
            try:
                for tag in soup.select(selector):
                    tag.decompose()
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")

        # 4. Breadcrumb elements
        for crumb in soup.find_all(class_=re.compile(r"breadcrumb", re.I)):
            crumb.decompose()

        # 5. Skip-to-content links (your exact logic)
        for a in soup.find_all("a"):
            text = a.get_text(strip=True).lower()
            if text.startswith(self.skip_link_prefixes):
                a.decompose()

        # 6. Hidden elements — display:none and visibility:hidden
        # Wrapped in try/except — find_all(style=True) can return nodes
        # without a proper .get() method (NavigableString, ProcessingInstruction, etc.)
        for tag in soup.find_all(style=True):
            try:
                attrs = getattr(tag, "attrs", None)
                if attrs is None:
                    continue
                style = (attrs.get("style") or "").replace(" ", "").lower()
                if "display:none" in style or "visibility:hidden" in style:
                    tag.decompose()
            except Exception:
                continue

        # 7. Remove img tags — already captured in Phase 1
        for img in soup.find_all("img"):
            img.decompose()

        # 8. Remove tags that became empty after all the above
        # (avoids cluttering markdown with orphaned empty divs/spans)
        changed = True
        while changed:
            changed = False
            for tag in soup.find_all(["div", "span", "p", "li", "ul", "ol", "section"]):
                if tag.get_text(strip=True) == "" and not tag.find(True):
                    tag.decompose()
                    changed = True

    def _to_markdown(self, soup: BeautifulSoup) -> str:
        """
        Convert cleaned soup to Markdown then post-process.
        Falls back to plain text if markdownify not installed.
        """
        if MARKDOWNIFY_AVAILABLE:
            raw_md = md(
                str(soup),
                heading_style="ATX",        # # H1  ## H2  etc.
                bullets="-",                # consistent bullet char
                strip=["a"],               # keep link text, drop [text](url) syntax
                                            # (links are captured separately)
                newline_style="backslash",
            )
            return self._clean_markdown(raw_md)
        else:
            text = soup.get_text(separator="\n", strip=True)
            return self._clean_plaintext(text)

    def _clean_markdown(self, text: str) -> str:
        """
        Post-process markdownify output.
        Your exact regex pipeline + a few extra cleanup passes.
        """
        # 1. Collapse 3+ blank lines → 2
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # 2. Remove empty headings (# with nothing / only spaces after)
        text = re.sub(r"^#{1,6}\s*$", "", text, flags=re.MULTILINE)

        # 3. Remove headings that are only punctuation/symbols
        text = re.sub(r"^#{1,6}\s*[\W_]+\s*$", "", text, flags=re.MULTILINE)

        # 4. Collapse multiple spaces/tabs within a line
        text = re.sub(r"[ \t]+", " ", text)

        # 5. Strip trailing whitespace per line (your exact logic)
        text = "\n".join(line.rstrip() for line in text.splitlines())

        # 6. Remove lines that are only markdown horizontal rules
        text = re.sub(r"^[-=*_]{3,}\s*$", "", text, flags=re.MULTILINE)

        # 7. Remove broken empty markdown links
        text = re.sub(r"\[\s*\]\([^)]*\)", "", text)

        # Remove "Click here" standalone lines
        text = re.sub(r"^Click here\s*$", "", text, flags=re.MULTILINE)

        # Remove "View all" standalone lines  
        text = re.sub(r"^View all\s*$", "", text, flags=re.MULTILINE)

        # Remove "Read more about ..." lines
        text = re.sub(r"^Read more about .+$", "", text, flags=re.MULTILINE)

        # Remove "और पढ़े" / "Read more" standalone lines
        text = re.sub(r"^(Read more|और पढ़े)\s*$", "", text, flags=re.MULTILINE)

        # Remove "Subscribe to" standalone
        text = re.sub(r"^Subscribe to\s*$", "", text, flags=re.MULTILINE)

        # Remove pagination blocks
        text = re.sub(r"(Current page|Next page|Last page|Next ›|Last »).*", "", text, flags=re.MULTILINE)

        # Remove "Published Date: DD-MM-YYYY" lines
        text = re.sub(r"^\s*Published Date:\s*\d{1,2}-\d{2}-\d{4}\s*$", "", text, flags=re.MULTILINE)

        # Remove lines that are only hashtags
        text = re.sub(r"^#\w+(\s+#\w+)*\s*$", "", text, flags=re.MULTILINE)

        # Remove lines that are pure whitespace
        lines = [line for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        return text.strip()

    def _clean_plaintext(self, text: str) -> str:
        """Fallback cleaner when markdownify is not installed."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()