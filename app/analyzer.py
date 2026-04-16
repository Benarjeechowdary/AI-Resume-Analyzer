from langchain_core.prompts import PromptTemplate

PROMPT = PromptTemplate(
    template="""
You are an expert technical recruiter.

Evaluate the following resume for the job role: {job_role}

Resume:
{resume_text}

IMPORTANT:
- Return ONLY valid JSON
- Each list must have only 3 points

Format:
{{
  "score": integer (0-95),
  "strengths": [],
  "weaknesses": [],
  "missing": [],
  "suggestions": []
}}
""",
    input_variables=["job_role", "resume_text"]
)

def analyze_chunks(chunks, job_role, llm):
    results = []

    for chunk in chunks:
        prompt = PROMPT.format(job_role=job_role, resume_text=chunk)
        try:
            res = llm.invoke(prompt)
            results.append(res)
        except Exception as e:
            print(f"Chunk failed: {e}")

    return results