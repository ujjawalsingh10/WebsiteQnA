from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import HUGGINGFACE_EMBEDDING_MODEL
import torch

logger = get_logger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
        
    try:
        if _embedding_model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(
                f"Initializing HuggingFace model: "
                f"{HUGGINGFACE_EMBEDDING_MODEL} on {device}"
            )

            _embedding_model = HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL,
                model_kwargs={'device' : device}
            )

            logger.info("HuggingFace embedding model loaded successfully")
            
        return _embedding_model

    except Exception as e:
        custom_error = CustomException(
            message="Error occurred while loading embedding model",
            error_detail=e,
        )
        logger.exception(custom_error)
        raise custom_error