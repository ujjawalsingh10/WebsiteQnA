import os
from dotenv import load_dotenv

load_dotenv()

# ********************** Chunking config ***************
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 50 ## Discard chunks smaller than this


#*************************** Embeddings **********************
# HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MPNet-base-v2"
EMBEDDING_BATCH_SIZE = 32   # MPNet supports larger batches comfortably


# ***************************** Qdrant ******************
COLLECTION_NAME  = "Website_qna_cluster"
DENSE_VECTOR_DIM = 768 ## embedding dim size    
QDRANT_BATCH_SIZE = 50    


# ******************* Retriever config *********************
CANDIDATE_COUNT  = 20     # retrieve this many before reranking
RERANK_TOP_K     = 5      # return this many after reranking
MIN_SCORE        = 0.3    # discard candidates below this score
# Cross-encoder model
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HF_TOKEN = os.getenv('HF_TOKEN')

## LLMs
HUGGINGFACE_REPO_ID = 'meta-llama/Llama-3.1-8B-Instruct'
# HUGGINGFACE_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
# HUGGINGFACE_REPO_ID = 'tiiuae/falcon-7b-instruct'
# HUGGINGFACE_REPO_ID = 'meta-llama/Llama-3.2-3B-Instruct'

