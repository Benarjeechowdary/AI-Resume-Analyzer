from app.loader import load_document
from app.splitter import split_text
from app.llm import get_llm
from app.analyzer import analyze_chunks
from app.aggregator import aggregate_results, compute_final_score
from app.embeddings import get_embedding_score

def main():
    file_path = "data/Benarjee_nalluri.pdf"
    job_role = "ML Engineer"

    print("Loading document...")
    resume_text = load_document(file_path)

    print("Splitting text...")
    chunks = split_text(resume_text)

    print("Loading LLM...")
    llm = get_llm()

    print("Analyzing chunks...")
    results = analyze_chunks(chunks, job_role, llm)

    if not results:
        raise Exception("No valid LLM responses")

    print("Aggregating results...")
    final = aggregate_results(results)

    print("Computing embedding score...")
    embedding_score = get_embedding_score(resume_text, job_role)

    final_score = compute_final_score(final, embedding_score)
    final["score"] = final_score

    print("\n✅ FINAL RESULT:\n")
    print(final)


if __name__ == "__main__":
    main()