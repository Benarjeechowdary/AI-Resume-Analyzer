from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


embed_model="sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL="llama-3.3-70b-versatile"

CHUNK_SIZE=500
CHUNK_OVERLAP=50