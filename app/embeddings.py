from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from app.config import embed_model

embed_model = HuggingFaceEmbeddings(model_name=embed_model)

def get_embedding_score(resume_text: str, job_role: str) -> float:
    emb_text = embed_model.embed_query(resume_text)
    emb_role = embed_model.embed_query(job_role)

    score = cosine_similarity([emb_text], [emb_role])[0][0]
    return score