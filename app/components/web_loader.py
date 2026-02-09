import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def parsing_url():
    try:
        
url = 'https://www.chrismytton.com/plain-text-websites/'
loader = WebBaseLoader(url)

docs = loader.load()
print(len(docs))