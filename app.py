import streamlit as st
import pandas as pd
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI ----------------
st.set_page_config(page_title="AI Resume Analyzer")

st.title("🔥 AI Resume Analyzer")
st.write("Upload your resume and get job match instantly!")

st.sidebar.title("📌 About")
st.sidebar.write("AI tool to analyze resume and suggest skills.")

# ---------------- FUNCTIONS ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_skills(text):
    skills = ['python','sql','machine learning','deep learning',
              'excel','power bi','tableau','communication','aws']
    
    found = []
    for skill in skills:
        if skill in text:
            found.append(skill)
    return found

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:

    with st.spinner("Analyzing your resume..."):

        text = extract_text(uploaded_file)
        clean = clean_text(text)

        resume_skills = extract_skills(clean)

        st.success("✅ Analysis Complete!")

        st.markdown("### ✅ Your Skills")
        st.write(", ".join(resume_skills))

        # ---------------- LOAD DATA ----------------
        df = pd.read_csv("Job_Description_Results_Total.csv")
        df['cleaned'] = df['Job Description'].apply(clean_text)

        # ---------------- ML MATCHING ----------------
        vectorizer = TfidfVectorizer()
        job_vectors = vectorizer.fit_transform(df['cleaned'])
        resume_vector = vectorizer.transform([clean])

        similarity = cosine_similarity(resume_vector, job_vectors)
        df['score'] = similarity[0]

        top_jobs = df.sort_values(by='score', ascending=False).head(1)

        job_text = top_jobs.iloc[0]['cleaned']
        job_skills = extract_skills(job_text)

        # ---------------- SCORE ----------------
        score = len(set(resume_skills) & set(job_skills)) / len(job_skills) * 100

        st.markdown("### 🎯 Match Score")
        st.progress(int(score))
        st.write(f"{score:.2f}% match")

        # ---------------- MISSING SKILLS ----------------
        missing = list(set(job_skills) - set(resume_skills))

        st.markdown("### 🔥 Skills You Need")
        for skill in missing:
         st.error(skill)

        # ---------------- TOP JOBS ----------------
        st.markdown("### 💼 Top Job Matches")
        top5 = df.sort_values(by='score', ascending=False).head(5)
        st.dataframe(top5[['Job Title','score']])
        # ---------------- REPORT ----------------
if uploaded_file:

    # your existing code
    text = extract_text(uploaded_file)
    clean = clean_text(text)

    resume_skills = extract_skills(clean)

    # scoring etc...
    score_percent = top_jobs.iloc[0]['score'] * 100

    # ---------------- REPORT ----------------
    report = f"""
Skills: {resume_skills}

Match Score: {score_percent:.2f}%

Missing Skills: {missing}
"""

    st.download_button("📥 Download Report", report)