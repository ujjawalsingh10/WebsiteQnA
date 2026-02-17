from langchain_core.prompts import PromptTemplate
from app.components.vector_store import load_vector_store
from app.components.llm import load_llm
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from langchain_core.output_parsers import StrOutputParser
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)

class RAGService:
    def __init__(self):
        try:
            logger.info('Initializing RAG Serive !')

            self.db = load_vector_store()
            self.llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

            self.prompt = PromptTemplate(
                template="""
                You are a technical assistant helping analyze website content.

                Use ONLY the provided context.
                You may infer the website name from titles, headers, or visible branding.

                If the answer cannot be determined from the context, say:
                "I don't have enough information."

                Context:
                {context}

                Question:
                {question}

                Answer:
                """,
                input_variables=["context", "question"]

            )

            self.parser = StrOutputParser()
            self.chain = self.prompt | self.llm | self.parser

            logger.info('RAG Serive initiated Successfully !')
        
        except Exception as e:
            custom_error = CustomException(
                message="Failed to initialize RAG service",
                error_detail=e
            )
            logger.exception(custom_error)
            raise custom_error
        

    def ask(self, question: str, debug: bool = False):
        try:
            logger.info(f"Processing query: {question}")

            retriever = self.db.as_retriever(
                search_type='similarity',
                search_kwargs = {'k': 6}
            )

            retrieved_docs = retriever.invoke(question)

            context_text = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            if debug == True:
                logger.info(f'Debug mode enabled -  Showing retrieved chunks !!\nQuery:{question}\n')
                for i, doc in enumerate(retrieved_docs):
                    logger.info(f"Chunk {i+1}: {doc.page_content}")

            answer = self.chain.invoke(
                {
                    'context' : context_text,
                    'question' : question
                })
            
            logger.info('Query successfully processed')

            return {
                'answer' : answer,
                'sources' : [doc.metadata for doc in retrieved_docs]
            }
        
        except Exception as e:
            custom_error = CustomException(
                message='Error while processing query',
                error_detail=e
            )
            logger.exception(custom_error)
            raise custom_error
        
