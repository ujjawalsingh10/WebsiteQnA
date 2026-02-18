from app.components.vector_store import save_vector_store
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
# from app.components.web_loaders.web_base_loader import load_and_parse_url ## using webbase loader
# from app.components.web_loaders.playwright_loader import load_and_parse_url ## using playwright loader
from app.components.chunking import create_text_chunks
from app.components.crawled_data_ingestion import DocumentLoader


logger = get_logger(__name__)

class IngestionService:

    @staticmethod
    # def ingest(url: str):
    def ingest():

        try:
            # logger.info(f"Starting ingestion for URL: {url}")
            logger.info(f"Starting ingestion")
            # output = load_and_parse_url(url)
            output = DocumentLoader.ingest_from_crawled_data()
            # chunks = create_text_chunks([output])
            chunks = create_text_chunks(output)
            save_vector_store(chunks)

            logger.info("Website indexed successfully")

            return {"status": "Website indexed successfully"}

        except Exception as e:
            custom_error =  CustomException(
                message="Website ingestion failed",
                error_detail=e
            )
            logger.exception(custom_error)
            raise custom_error

# IngestionService.ingest()
# print('Done')