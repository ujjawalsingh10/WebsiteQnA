from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl, Field
from typing import Annotated, Any

from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.components.retriever import RAGService
from app.components.ingestion import IngestionService
from app.components.vector_store import save_vector_store, load_vector_store

logger = get_logger(__name__)

app = FastAPI()

rag_service = RAGService()

class IngestURL(BaseModel):
    """
    URL text to be extracted
    """
    url: Annotated[AnyUrl, Field(..., description='Enter a valid URL')]

class ChatRequest(BaseModel):
    question: Annotated[str, Field(..., min_length=4, max_length=200, description='Enter a query about the website')]


@app.post('/ingest')
def ingest(request: IngestURL):
    try:
        return IngestionService.ingest(str(request.url))
    
    except Exception as e:
        custom_error = CustomException(
            message='Ingestion Request failed !',
            error_detail=e
        )
        logger.error(custom_error)
        raise HTTPException(status_code=500, detail=custom_error.error_message)
    
@app.post('/chat')
def chat(request: ChatRequest):
    try:
        return rag_service.ask(request.question, debug=True)
    
    except Exception as e:
        custom_error = CustomException(
            message='Chat Request failed !',
            error_detail=e
        )
        logger.error(custom_error)
        raise HTTPException(status_code=500, detail=custom_error.error_message)
    

