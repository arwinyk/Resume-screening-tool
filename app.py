import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader

# Load trained model and vectorizer
with open("resume_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Resume Screening Tool")
st.write("Upload a resume (PDF or TXT) to predict the candidate category.")

# File upload
uploaded_file = st.file_uploader("Choose a resume", type=["pdf", "txt"])

def read_resume(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + " "
        return text
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return ""

if uploaded_file is not None:
    resume_text = read_resume(uploaded_file)
    st.write("Resume Content:")
    st.write(resume_text[:500] + "...")  # show first 500 chars

    # Transform and predict
    X = vectorizer.transform([resume_text])
    prediction = model.predict(X)[0]
    st.success(f"Predicted Category: **{prediction}**")
