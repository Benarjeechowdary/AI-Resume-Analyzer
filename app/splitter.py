from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE,CHUNK_OVERLAP

def split_text(docs):

    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    chunks=splitter.split_text(docs)
    return chunks
