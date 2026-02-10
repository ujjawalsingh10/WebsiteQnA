import os
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def fetch_url(url):
    if not url.startswith('http'):
        raise ValueError('URL must start with http:// or https://')
    logger.info(f"Loading URL: {url}")

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text