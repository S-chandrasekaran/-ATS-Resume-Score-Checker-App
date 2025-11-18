import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import re

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample skill list (you can expand this or load from a file)
SKILL_KEYWORDS = {
    "python", "java", "c++", "sql", "excel", "power bi", "tableau", "machine learning",
    "deep learning", "nlp", "data analysis", "data visualization", "communication",
    "problem solving", "teamwork", "leadership", "project management", "aws", "azure",
    "git", "docker", "kubernetes", "linux", "html", "css", "javascript"
}

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Clean and normalize text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Extract skills using keyword matching
def extract_skills(text):
    found_skills = set()
    for skill in SKILL_KEYWORDS:
        if skill in text:
            found_skills.add(skill)
    return found_skills

# Compute semantic similarity
def compute_similarity(resume_text, job_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    score = util.cos_sim(resume_embedding, job_embedding).item()
    return round(score * 100, 2)

# Streamlit UI
st.set_page_config(page_title="ATS Resume Score Checker", layout="centered")
st.title("🧠 ATS Resume Score Checker")
st.write("Upload your resume and paste a job description to get a match score.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description")

if uploaded_file and job_description:
    with st.spinner("Analyzing your resume..."):
        resume_text = clean_text(extract_text_from_pdf(uploaded_file))
        job_text = clean_text(job_description)

        score = compute_similarity(resume_text, job_text)
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)

        matched_skills = resume_skills & job_skills
        missing_skills = job_skills - resume_skills

    st.success("✅ Analysis Complete!")

    st.subheader(f"📊 Match Score: {score}%")

    st.markdown("### 🧩 Matched Skills")
    if matched_skills:
        st.write(", ".join(sorted(matched_skills)))
    else:
        st.write("No matched skills found.")

    st.markdown("### ❌ Missing Skills")
    if missing_skills:
        st.write(", ".join(sorted(missing_skills)))
    else:
        st.write("No missing skills detected.")

    with st.expander("📜 View Extracted Resume Text"):
        st.write(resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text)
