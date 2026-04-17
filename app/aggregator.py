from sklearn.metrics.pairwise import cosine_similarity
from app.config import embed_model
from langchain_huggingface import HuggingFaceEmbeddings

from collections import Counter


def aggregate_results(results,job_role):
    final = {
        "score": 0,
        "strengths": [],
        "weaknesses": [],
        "missing": [],
        "suggestions": []
    }

    for r in results:
        final["score"] += r.score
        final["strengths"].extend(r.strengths)
        final["weaknesses"].extend(r.weaknesses)
        final["missing"].extend(r.missing)
        final["suggestions"].extend(r.suggestions)

    final["score"] = int(final["score"] / len(results))
    emb_model = HuggingFaceEmbeddings(model_name=embed_model)
    job_embedding=emb_model.embed_query(job_role)

    for key in ["strengths", "weaknesses", "missing", "suggestions"]:
        items = final[key]

        # Step 2: take top frequent
        items = get_top_k(items, k=10)  # take more initially
        items = remove_similar_items(items, emb_model)  # remove similar ones

        # Step 3: rank by job relevance
        
        final[key] = rank_by_similarity(items, job_embedding, emb_model, 3)

    return final


def compute_final_score(final, embedding_score):
    return int((final["score"] * 0.7) + (embedding_score * 100 * 0.3))


def get_top_k(items, k=3):
    
    counter = Counter(items)
    return [item for item, _ in counter.most_common(k)]



def rank_by_similarity(items, job_embedding, emb_model, k=3):
    item_embeddings = emb_model.embed_documents(items)
    
    scores = cosine_similarity(item_embeddings, [job_embedding]).flatten()
    
    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
    
    return [item for item, _ in ranked[:k]]



def remove_similar_items(items, emb_model, threshold=0.80):
    embeddings = emb_model.embed_documents(items)
    
    filtered_items = []

    for i, emb in enumerate(embeddings):
        keep = True
        
        for kept_item in filtered_items:
            kept_emb = emb_model.embed_query(kept_item)
            sim = cosine_similarity([emb], [kept_emb])[0][0]
            
            if sim > threshold:
                keep = False
                break
        
        if keep:
            filtered_items.append(items[i])

    return filtered_items