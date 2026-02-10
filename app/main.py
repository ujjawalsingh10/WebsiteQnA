# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI(
#     title='Website QnA API',
#     version='0.0.1'
# )

# @app.get('/health')
# def health():
#     return {'status': 'ok'}

from app.components.web_loaders.web_base_loader import load_and_parse_url
from app.components.chunking import create_text_chunks
from app.components.embeddings import get_embedding_model
from app.components.vector_store import save_vector_store, load_vector_store

# def main():
#     model = get_embedding_model()
#     print("Embedding model ready:", model)
#     sentences = ["This is an example sentence", "Each sentence is converted"]
#     res = model.embed_documents(sentences)
#     print(res)

# if __name__ == "__main__":
#     main()

url = 'https://www.chrismytton.com/plain-text-websites/'

output = load_and_parse_url(url)
chunks = create_text_chunks([output])
db = save_vector_store(chunks)
loaded_db = load_vector_store()

print(db)

