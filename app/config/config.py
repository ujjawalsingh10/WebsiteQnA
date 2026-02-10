import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

HF_TOKEN = os.getenv('HF_TOKEN')
HUGGINGFACE_REPO_ID = 'meta-llama/Llama-3.1-8B-Instruct'
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = 'vectorstore/db_faiss'
# DATA_PATH ='data/'