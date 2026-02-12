import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


# version 2-----------------------
# import logging
# import os
# from logging.handlers import RotatingFileHandler
# from datetime import datetime
# from from_root import from_root


# # ==============================
# # Configuration Constants
# # ==============================

# LOG_DIR = "logs"
# MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
# BACKUP_COUNT = 3


# # ==============================
# # Setup Log File Path
# # ==============================

# log_dir_path = os.path.join(from_root(), LOG_DIR)
# os.makedirs(log_dir_path, exist_ok=True)

# log_file_name = f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
# log_file_path = os.path.join(log_dir_path, log_file_name)


# # ==============================
# # Logger Configuration
# # ==============================

# def configure_logger() -> None:
#     """
#     Configure root logging once.
#     Should be called only once at application startup.
#     """

#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)

#     # Prevent duplicate handlers (important for FastAPI reload)
#     if logger.handlers:
#         return

#     formatter = logging.Formatter(
#         "[ %(asctime)s ] [ %(levelname)s ] [ %(name)s ] - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S"
#     )

#     # Rotating file handler
#     file_handler = RotatingFileHandler(
#         log_file_path,
#         maxBytes=MAX_LOG_SIZE,
#         backupCount=BACKUP_COUNT,
#         encoding="utf-8"
#     )
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(formatter)

#     # Console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)


# def get_logger(name: str) -> logging.Logger:
#     """
#     Get a named logger for modules.
#     """
#     return logging.getLogger(name)


# # Call once at startup
# configure_logger()