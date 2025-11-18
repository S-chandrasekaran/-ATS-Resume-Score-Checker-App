import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import spacy

# Load models
import en_core_web_sm
nlp = en_core_web_sm.load()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract skills using spaCy
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["SKILL", "ORG", "PRODUCT"]]
    return list(set(skills))

# Compute similarity score
def compute_similarity(resume_text, job_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    score = util.cos_sim(resume_embedding, job_embedding).item()
    return round(score * 100, 2)

# Streamlit UI
st.title("🧠 ATS Resume Score Checker")
st.write("Upload your resume and paste a job description to get a match score.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    score = compute_similarity(resume_text, job_description)
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched_skills = set(resume_skills) & set(job_skills)
    missing_skills = set(job_skills) - set(resume_skills)

    st.subheader(f"✅ Match Score: {score}%")
    st.markdown("### 🧩 Matched Skills")
    st.write(", ".join(matched_skills) if matched_skills else "No matched skills found.")

    st.markdown("### ❌ Missing Skills")
    st.write(", ".join(missing_skills) if missing_skills else "No missing skills detected.")
