"""
Resume Parser — Streamlit Web App
==================================
Run with:  streamlit run app.py
"""

import sys
import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from resume_parser import ResumeParser
from scorer import CandidateScorer, rank_candidates

# Page config 
st.set_page_config(
    page_title="Resume Parser ATS",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume Parsing System")
st.markdown(
    "**NLP-powered ATS** — Upload resumes, paste a job description, "
    "and get ranked candidates instantly."
)
st.divider()

# Sidebar 
with st.sidebar:
    st.header("⚙️ Settings")
    required_years = st.number_input("Required Years of Experience", min_value=0.0, value=0.0, step=0.5)
    top_n = st.slider("Show Top N Candidates", 1, 50, 10)

# Job description input 
st.subheader("1. Job Description")
jd_text = st.text_area(
    "Paste the job description here",
    height=180,
    placeholder="We are looking for a Python developer with experience in Django, PostgreSQL, AWS…",
)

# Resume upload 
st.subheader("2. Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload one or more resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

# Run analysis 
if st.button("🚀 Analyze Resumes", type="primary", disabled=not (jd_text and uploaded_files)):

    parsed_resumes = []
    errors = []

    with st.spinner("Parsing resumes…"):
        for uf in uploaded_files:
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            try:
                rp = ResumeParser(tmp_path)
                p = rp.parse()
                p["_source_file"] = uf.name
                parsed_resumes.append(p)
            except Exception as e:
                errors.append(f"{uf.name}: {e}")

    if errors:
        st.warning("Some files could not be parsed:\n" + "\n".join(errors))

    if not parsed_resumes:
        st.error("No resumes parsed successfully.")
        st.stop()

    ranked = rank_candidates(parsed_resumes, jd_text, required_years or None)
    ranked = ranked[:top_n]

    # Summary table 
    st.divider()
    st.subheader("📊 Candidate Rankings")

    table_data = []
    for i, r in enumerate(ranked, 1):
        table_data.append({
            "Rank":           i,
            "Name":           r.get("name") or "Unknown",
            "Email":          r.get("email") or "—",
            "Score (%)":      r["total_score"],
            "Skill Match":    r["breakdown"]["skill_match"],
            "TF-IDF Sim":     r["breakdown"]["tfidf_sim"],
            "Recommendation": r["recommendation"],
        })

    df = pd.DataFrame(table_data)

    def color_score(val):
        if val >= 75:
            return "background-color: #d4edda; color: #155724"
        elif val >= 55:
            return "background-color: #fff3cd; color: #856404"
        else:
            return "background-color: #f8d7da; color: #721c24"

    styled = df.style.applymap(color_score, subset=["Score (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Individual cards 
    st.divider()
    st.subheader("🔍 Detailed Analysis")

    for i, r in enumerate(ranked, 1):
        with st.expander(f"#{i}  {r.get('name', 'Unknown')}  —  {r['total_score']}%"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Score", f"{r['total_score']}%")
                st.write(f"**Recommendation:** {r['recommendation']}")
                st.write(f"**Email:** {r.get('email', '—')}")
                st.write("**Matched Skills:**")
                if r["matched_skills"]:
                    st.success(", ".join(r["matched_skills"]))
                else:
                    st.info("No skills matched")
            with col2:
                st.write("**Score Breakdown:**")
                breakdown_df = pd.DataFrame(
                    [{"Component": k.replace("_", " ").title(), "Score (%)": v}
                     for k, v in r["breakdown"].items()]
                )
                st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
                st.write("**Missing Skills:**")
                if r["missing_skills"]:
                    st.error(", ".join(r["missing_skills"]))
                else:
                    st.success("All required skills matched!")

    # Download 
    st.divider()
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps(ranked, indent=2, ensure_ascii=False),
        file_name="ranked_candidates.json",
        mime="application/json",
    )
