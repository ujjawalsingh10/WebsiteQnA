# import os
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# from app.common.logger import get_logger
# from app.common.custom_exceptions import CustomException

# logger = get_logger(__name__)

# class DocumentLoader:

#     @staticmethod
#     def ingest_from_crawled_data():
#         try:
#             logger.info('Starting ingestion from crawled data')

#             text_loader = DirectoryLoader(
#                 'data/raw/pages',
#                 glob="**/*.md",
#                 loader_cls=TextLoader,
#                 loader_kwargs={
#                     'encoding' : 'utf-8',
#                     'errors' : 'ignore'
#                 },
#                 show_progress=True
#             )
#             text_docs = text_loader.load()
#             pdf_loader = DirectoryLoader(
#                 'data/raw/pdfs',
#                 glob="**/*.pdf",
#                 loader_cls=PyPDFLoader,
#                 show_progress=True
#             )

#             pdf_docs = pdf_loader.load()

#             documents = text_docs + pdf_docs

#             logger.info(f'Total Documents loaded: {len(documents)}')

#             logger.info("Crawled website indexed successfully")

#             return documents
        
#         except Exception as e:
#             custom_error = CustomException(
#                 message="Crawled data ingestion failed",
#                 error_detail=e
#             )
#             logger.exception(custom_error)
#             raise custom_error

import os
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)


class DocumentLoader:

    @staticmethod
    def ingest_from_crawled_data(folder_path: str = "data/raw"):

        try:
            logger.info("Starting ingestion from crawled data")

            supported_extensions = (".md", ".txt", ".pdf")

            # ---- Collect all supported files first ----
            all_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(supported_extensions):
                        all_files.append(os.path.join(root, file))

            logger.info(f"Found {len(all_files)} supported files")

            documents = []

            # ---- Progress Bar ----
            with tqdm(total=len(all_files), desc="Ingesting files", unit="file") as pbar:

                for file_path in all_files:
                    try:
                        pbar.set_postfix_str(os.path.basename(file_path))

                        # ---- TEXT / MARKDOWN ----
                        if file_path.endswith((".md", ".txt")):
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                text = f.read()

                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": file_path,
                                        "type": "text"
                                    }
                                )
                            )

                        # ---- PDF ----
                        elif file_path.endswith(".pdf"):
                            pdf_loader = PyPDFLoader(file_path)
                            pdf_docs = pdf_loader.load()

                            for doc in pdf_docs:
                                doc.metadata["source"] = file_path
                                doc.metadata["type"] = "pdf"

                            documents.extend(pdf_docs)

                    except Exception as file_error:
                        logger.error(f"Skipping file: {file_path} | Error: {file_error}")

                    finally:
                        pbar.update(1)

            logger.info(f"Loaded {len(documents)} documents successfully")

            return documents

        except Exception as e:
            custom_error = CustomException(
                message="Crawled data ingestion failed",
                error_detail=e
            )
            logger.exception(custom_error)
            raise custom_error