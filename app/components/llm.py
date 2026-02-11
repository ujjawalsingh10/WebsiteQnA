from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info('Loading LLM from Huggingface')
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            task='text-generation',
            huggingfacehub_api_token=HF_TOKEN,
            max_new_tokens=256,
            temperature=0.2
        )

        llm = ChatHuggingFace(llm=llm_endpoint)
        logger.info('LLM loaded successfully...')
        return llm
    
    except Exception as e:
        custom_error = CustomException(
            message="Failed to load the LLM",
            error_detail=e
        )
        logger.exception(custom_error)
        raise custom_error

