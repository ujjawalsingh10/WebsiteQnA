from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)

def create_text_chunks(documents):
    """
    Split documents into overlapping text chunks using RecursiveCharacterTextSplitter.
    """
    if not documents:
        raise CustomException(message='No documents were found !')
    
    logger.info(f"Splitting {len(documents)} documents into chunks")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP
        )

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks")
        return text_chunks
    
    except Exception as e:
        custom_error = CustomException(
            message='Failed to generate chunks',
            error_detail=e
        )
        logger.exception(str(custom_error))
        raise custom_error
