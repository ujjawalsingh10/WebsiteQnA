import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 150
DB_FAISS_PATH = 'vectorstore/db_faiss'

HF_TOKEN = os.getenv('HF_TOKEN')

## LLMs
HUGGINGFACE_REPO_ID = 'meta-llama/Llama-3.1-8B-Instruct'
# HUGGINGFACE_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
# HUGGINGFACE_REPO_ID = 'tiiuae/falcon-7b-instruct'
# HUGGINGFACE_REPO_ID = 'meta-llama/Llama-3.2-3B-Instruct'


## Embeddings
# HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MPNet-base-v2"