from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)

def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException(message='No text chunks were found')
        
        logger.info('Generating new vectorstore...')

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info('Saving vectorstore')

        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        db.save_local(DB_FAISS_PATH)

        logger.info('VectorStore saved successfully...')

        return db
    
    except Exception as e:
        custom_error = CustomException(
            message='Failed to create the vector store',
            error_detail=e
        )
        logger.exception(custom_error)
        raise custom_error


def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info('Loading existing vectorstore..')
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(f'FAISS Index not found at {DB_FAISS_PATH}')
    
    except Exception as e:
        custom_error = CustomException(
            message="Failed to load vector store",
            error_detail=e
        )
        logger.exception(custom_error)
        raise custom_error
    