import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from markdownify import markdownify as md
from hashlib import md5
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.components.crawler.config import CrawlerConfig
from app.components.crawler.utils import normalize_url, is_internal_link
from collections import deque

logger = get_logger(__name__)

class WebCrawler:
    """
    Crawler used to extract text, images, pdfs from website
    Features:
    - Depth-based crawling
    - Internal domain restriction
    - HTML -> Markdown conversion
    - PDF download
    - Image download
    - Duplicate prevention
    - Retry mechanism
    - Structured logging
    """

    def __init__(self, BASE_URL):
        """
        :param base_url: Root website URL to crawl
        """
        self.BASE_URL = BASE_URL.rstrip('/')
        self.domain = urlparse(BASE_URL).netloc

        self.visited = set()
        self.queue = deque([(self.BASE_URL, 0)])  #(url, depth)
        self.queued = set([self.BASE_URL])

        self.headers = {
            'User-Agent' : CrawlerConfig.USER_AGENT
        }

        self.setup_storage()
    
    def setup_storage(self):
        """
        Create required folder structure for storing:
        - Markdown text
        - PDFs
        - Images
        """
        base = CrawlerConfig.BASE_STORAGE_PATH

        self.text_path = os.path.join(base, CrawlerConfig.TEXT_FOLDER)
        self.pdf_path = os.path.join(base, CrawlerConfig.PDF_FOLDER)
        self.image_path = os.path.join(base, CrawlerConfig.IMAGE_FOLDER)


        os.makedirs(self.text_path, exist_ok=True)
        os.makedirs(self.pdf_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

        logger.info("Storage directories initialized")

    def fetch(self, url):
        """
        Fetch URL with retry mechanism.

        :param url: URL to fetch
        :return: requests.Response or None
        """
        
        for attempt in range(1, CrawlerConfig.MAX_RETRIES + 1):
            try:
                logger.info(f"[Attempt {attempt}] Fetching URL: {url}")

                response = requests.get(
                    url, 
                    headers=self.headers,
                    timeout=CrawlerConfig.REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    return response
                elif response.status_code == 404:
                    logger.warning(f"404 Not found: {url} !")
                    return None
                else:
                    logger.warning(
                        f"Non-200 response ({response.status_code}) for {url}"
                    )

            except requests.RequestException as e:
                custom_error = CustomException(
                    message='Request failed for URL: {url}',
                    error_detail=e
                )
                logger.exception(custom_error)
                time.sleep(1)

        logger.error(f'Max retries exceeded for {url}')        
        return None
    
    def generate_id(self, url):
        """
        Generate deterministic MD5 hash for filename.

        :param url: Source URL
        :return: md5 hash string
        """
        return md5(url.encode('utf-8')).hexdigest()
    
    def process_content(self, html_content, url):
        """
        Convert HTML to Markdown and store locally.

        Removes noisy tags to improve embedding quality.

        :param html_content: Raw HTML
        :param url: Source URL
        """

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            logger.info(f'Parsing URL: {url}')

            ## Removing noisy elements that confuse AI
            for noise in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                noise.decompose()
            
            ## converting the remaining cleaned HTML to markdown (Preserves Tables)
            markdown_text = md(str(soup), heading_style='ATX')

            file_id = self.generate_id(url)
            save_path = os.path.join(self.text_path, f"{file_id}.md")

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"--- SOURCE: {url} --- \n\n")
                f.write(markdown_text)
            
            logger.info(f"Saved processed content to {save_path}")

        except Exception as e:
            custom_error = CustomException(
                message=f"Error processing HTML for url {url}",
                error_detail=e
            )
            logger.exception(custom_error)


    def download_file(self, url, folder):
        """
        Download and store binary files (PDFs or images).

        :param url: File URL
        :param folder: Target folder path
        """
        try:
            file_id = self.generate_id(url)
            ext = os.path.splitext(urlparse(url).path)[1]
            if not ext:
                logger.warning(f"Skipping file without extension: {url}")
                return 
            
            save_path = os.path.join(folder, f"{file_id}{ext}")

            if os.path.exists(save_path):
                logger.debug(f"File already exists: {save_path}")
                return 

            response = self.fetch(url)
            if response:
                with open(save_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Download file: {url}")
        
        except Exception as e:
            custom_error = CustomException(
                message=f"Failed downloading URL: {url}", 
                error_detail=e
            )
            logger.exception(custom_error)


    def crawl(self):
        """
        Breadth-First Search crawl with:
        - Depth control
        - Page limit
        - Internal domain restriction
        """

        logger.info(f"Staring crawl: {self.BASE_URL}")

        while self.queue and len(self.visited) < CrawlerConfig.MAX_PAGES:
            current_url, depth = self.queue.popleft()   

            # depth check
            if depth > CrawlerConfig.MAX_DEPTH:
                logger.debug(f"Max depth reached at {current_url}")
                continue
            
            # already visited
            if current_url in self.visited:
                continue

            response = self.fetch(current_url)
            if not response:
                continue

            self.visited.add(current_url)

            content_type = response.headers.get('Content-Type', '')

            ## HTML processing
            if 'text/html' in content_type:
                logger.info(f"Processing HTML: {current_url}")
                self.process_content(response.text, current_url)

                soup = BeautifulSoup(response.content, 'html.parser')

                ### extract links
                for tag in soup.find_all('a', href=True):
                    normalized = normalize_url(current_url, tag['href'])

                    if normalized.endswith('.pdf'):
                        self.download_file(normalized, self.pdf_path)

                    elif is_internal_link(self.domain, normalized):
                        if normalized not in self.visited and normalized not in self.queued:
                            self.queue.append((normalized, depth + 1))
                            self.queued.add(normalized)
                
                ## extract images
                for img in soup.find_all("img", src=True):
                    img_url = normalize_url(current_url, img['src'])

                    if any(img_url.lower().endswith(ext)
                           for ext in CrawlerConfig.IMAGE_EXTENSIONS):
                        self.download_file(img_url, self.image_path)
            
            elif 'application/pdf' in content_type:
                logger.info(f"Processing direct PDF: {current_url}")
                self.download_file(current_url, self.pdf_path)
            
            time.sleep(CrawlerConfig.REQUEST_DELAY)
        
        logger.info(f'Crawl Finished ! Total pages visited: {len(self.visited)}')