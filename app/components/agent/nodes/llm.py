from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(temperature=0.1):
    """
    returns llm on function call
    """
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )