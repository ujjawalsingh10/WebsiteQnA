from langchain_community.document_loaders import WebBaseLoader
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.utilities import remove_tag_lines, remove_navigation, normalize_whitespace
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = get_logger(__name__)

def preprocess_url_page_content(page_content: str) -> str:
    """
    clean raw page text extracted by webbase loader
    """
    if not page_content:
        return ""
    
    logger.debug("Preprocessing the url page content")

    text = normalize_whitespace(page_content)
    text = remove_navigation(text)
    text = remove_tag_lines(text)

    logger.debug("Page content successfully preprocessed")
    return text

def load_and_parse_url(url: str) -> Document:
    """
    load a URL using WebBaseLoader and return cleaned content + metadata
    :output - Document metadata - source, title, description |  page_content
    """

    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    logger.info(f"Loading URL: {url}")

    try:            
        loader = WebBaseLoader(url)
        documents = loader.load()

        if not documents:
            raise CustomException(message='Web Loader returned no documents')
        
        logger.info(f"Successfully fetched {len(documents)} documents")
        docs = documents[0]
        cleaned_text = preprocess_url_page_content(docs.page_content)
        
        with open('parsed_site_text.txt', 'w', encoding='utf-8') as f:
            print('Writing website content in text file')
            f.write(cleaned_text)

        return Document(
            page_content=cleaned_text,
            metadata={
                'source': docs.metadata.get('source', url),
                'title' : docs.metadata.get('title'),
                'description': docs.metadata.get('description')
            }
        )
    
        # result['source'] = docs.metadata.get('source', url)
        # result['title'] = docs.metadata.get('title')
        # result['description'] = docs.metadata.get('description')
        # result['page_content'] = preprocess_url_page_content(docs.page_content)

        # return result
    
    except Exception as e:
        custom_error = CustomException(
            message='Failed to load website',
            error_detail=e
        )
        logger.exception(str(custom_error))
        raise custom_error

