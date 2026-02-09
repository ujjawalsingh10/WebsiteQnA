from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title='Website QnA API',
    version='0.0.1'
)

@app.get('/health')
def health():
    return {'status': 'ok'}