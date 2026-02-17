from langchain_core.documents import Document
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.utilities import remove_tag_lines, remove_navigation, normalize_whitespace

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re

logger = get_logger(__name__)


def preprocess_url_page_content(page_content: str) -> str:
    """
    Clean raw page text extracted by Playwright
    """
    if not page_content:
        return ""

    logger.debug("Preprocessing the url page content")

    text = normalize_whitespace(page_content)
    text = remove_navigation(text)
    text = remove_tag_lines(text)

    logger.debug("Page content successfully preprocessed")
    return text


def extract_main_content(html: str) -> str:
    """
    Extract semantic main content from documentation pages.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious noise
    for selector in [
        "nav", "header", "footer", "aside",
        ".sidebar", ".toc", ".navbar", ".menu"
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Target semantic containers
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
    )

    if main_content:
        text = main_content.get_text(separator="\n")
    else:
        logger.warning("Main content container not found. Using full page.")
        text = soup.get_text(separator="\n")

    # Normalize whitespace early
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def load_and_parse_url(url: str) -> Document:
    """
    Load a URL using Playwright and return cleaned content + metadata.
    """

    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    logger.info(f"Loading URL with Playwright: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            html = page.content()
            title = page.title()

            browser.close()

        if not html:
            raise CustomException(message="Playwright returned empty HTML")

        logger.info("Successfully fetched page HTML")

        extracted_text = extract_main_content(html)
        cleaned_text = preprocess_url_page_content(extracted_text)

        with open("parsed_site_text.txt", "w", encoding="utf-8") as f:
            logger.info("Writing website content to parsed_site_text.txt")
            f.write(cleaned_text)

        return Document(
            page_content=cleaned_text,
            metadata={
                "source": url,
                "title": title,
                "description": None
            }
        )

    except Exception as e:
        custom_error = CustomException(
            message="Failed to load website using Playwright",
            error_detail=e
        )
        logger.exception(str(custom_error))
        raise custom_error