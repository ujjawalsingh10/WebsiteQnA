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

                When you use information from a context block, cite it using its bracket number like [1], [2].

                Context:
                {context}

                Question:
                {question}

                Answer (with citations):
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
                # search_type='mmr',
                search_kwargs = {'k': 6}
            )

            retrieved_docs = retriever.invoke(question)

            context_blocks = []
            references = []

            for idx, doc in enumerate(retrieved_docs):
                ref_number = idx + 1

                context_blocks.append(
                    f"[{ref_number}] {doc.page_content}"
                )

                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', None)

                filename = source.split('\\')[-1]

                if page is not None:
                    references.append(f"[{ref_number}] {filename} (Page {page})")
                else:
                    references.append(f"[{ref_number}] {filename}")
                
            context_text = "\n\n".join(context_blocks)


            # context_text = "\n\n".join(
            #     [doc.page_content for doc in retrieved_docs]
            # )

            if debug == True:
                logger.info(f'Debug mode enabled -  Showing retrieved chunks !!\nQuery:{question}\n')
                for i, doc in enumerate(retrieved_docs):
                    logger.info(f"Chunk {i+1}: {doc.page_content}")
                logger.info(f"--------------------------------------------------------")
            logger.info(f"This is the context text : {context_text}")

            answer = self.chain.invoke(
                {
                    'context' : context_text,
                    'question' : question
                })
            
            logger.info('Query successfully processed')
            
            final_answer = """
            {answer}
            Sources : {references}
            """
            return {
                'answer' : answer,
                'references' : list(set(references))
            }

            # return {
            #     'asnwer' : final_answer
            # }     
           
        except Exception as e:
            custom_error = CustomException(
                message='Error while processing query',
                error_detail=e
            )
            logger.exception(custom_error)
            raise custom_error
        
