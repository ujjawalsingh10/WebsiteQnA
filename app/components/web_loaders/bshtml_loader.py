import os
import requests
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP
# from bs4 import BeautifulSoup

logger = get_logger(__name__)

def fetch_url(url):
    print('Fetching....')
    if not url.startswith('http'):
        raise ValueError('URL must start with http:// or https://')
    # logger.info(f"Loading URL: {url}")

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    # soup = BeautifulSoup(response.content, 'html.parser')
    loader = BSHTMLLoader(url)
    docs = loader.load()
    with open('bstml_content.txt', 'w', encoding='utf-8') as f:
        f.write(docs[0].page_content)
    # return response.text
    print('Done!')

url = 'https://reference.langchain.com/python/langchain_core/retrievers/'

result = fetch_url(url)


# print(result)