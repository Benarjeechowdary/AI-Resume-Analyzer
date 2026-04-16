import streamlit as st
import os

from app.loader import load_document
from app.splitter import split_text
from app.llm import get_llm
from app.analyzer import analyze_chunks
from app.aggregator import aggregate_results, compute_final_score
from app.embeddings import get_embedding_score

st.set_page_config(page_title="AI Resume Optimizer", layout="wide")

st.title("🚀 AI Resume Optimizer")

# Folder to store uploads
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

# UI Inputs
uploaded_file = st.file_uploader("📄 Upload your Resume", type=["pdf", "docx"])
job_role = st.text_input("💼 Enter Job Role (e.g., ML Engineer)")

analyze_btn = st.button("Analyze Resume")

# ------------------------------
# MAIN LOGIC
# ------------------------------

if analyze_btn:
    if uploaded_file is None or job_role.strip() == "":
        st.warning("Please upload resume and enter job role")
    else:
        # Save file
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Processing... Please wait ⏳")

        try:
            # Step 1: Load
            resume_text = load_document(file_path)

            # Step 2: Split
            chunks = split_text(resume_text)

            # Step 3: LLM
            llm = get_llm()

            # Step 4: Analyze
            results = analyze_chunks(chunks, job_role, llm)

            if not results:
                st.error("Analysis failed. Try again.")
                st.stop()

            # Step 5: Aggregate
            final = aggregate_results(results,job_role)

            # Step 6: Embedding score
            embedding_score = get_embedding_score(resume_text, job_role)

            final_score = compute_final_score(final, embedding_score)
            final["score"] = final_score

            # ------------------------------
            # DISPLAY RESULTS
            # ------------------------------

            st.success("✅ Analysis Complete")

            st.metric("📊 Resume Score", f"{final['score']} / 100")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💪 Strengths")
                for s in final["strengths"]:
                    st.write(f"✔ {s}")

                st.subheader("⚠ Weaknesses")
                for w in final["weaknesses"]:
                    st.write(f"❌ {w}")

            with col2:
                st.subheader("📉 Missing Skills")
                for m in final["missing"]:
                    st.write(f"📌 {m}")

                st.subheader("🚀 Suggestions")
                for s in final["suggestions"]:
                    st.write(f"👉 {s}")

        except Exception as e:
            st.error(f"Error: {str(e)}")