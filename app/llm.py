from langchain_groq import ChatGroq
from app.config import GROQ_MODEL
from app.schemas import Resume

def get_llm():
    model = ChatGroq(model=GROQ_MODEL)
    return model.with_structured_output(Resume)