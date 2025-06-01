import streamlit as st
import joblib
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load model & vectorizer & label encoder
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.strip()
    return text

def preprocess(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

# Streamlit UI
st.set_page_config(page_title="Cyberbullying Classifier", page_icon="💬")
st.title("💬 Deteksi Cyberbullying Komentar Teks")

with st.form("form"):
    user_input = st.text_area("Masukkan komentar:", height=150)
    submitted = st.form_submit_button("Prediksi")

if submitted:
    if user_input.strip() == "":
        st.warning("Masukkan komentar terlebih dahulu.")
    else:
        preprocessed_text = preprocess(user_input)
        tfidf_vector = vectorizer.transform([preprocessed_text])
        prediction = model.predict(tfidf_vector)
        label = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Hasil Prediksi:")
        if label.lower() == "non-cyberbullying":
            st.success("✅ Komentar ini **tidak mengandung cyberbullying.**")
        else:
            st.error("⚠️ Komentar ini **mengandung unsur cyberbullying!**")
