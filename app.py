import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load model dan alat bantu
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Fungsi pembersihan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Fungsi preprocessing sederhana
def preprocess(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

# Tampilan Streamlit
st.set_page_config(page_title="Cyberbullying Classifier", page_icon="💬")
st.title("💬 Deteksi Jenis Komentar Cyberbullying")

with st.form("form"):
    user_input = st.text_area("Masukkan komentar atau kalimat:", height=150)
    submitted = st.form_submit_button("Prediksi")

if submitted:
    if user_input.strip() == "":
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
    else:
        preprocessed_text = preprocess(user_input)
        vectorized = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized)
        label = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Hasil Prediksi:")
        if label == "not_cyberbullying":
            st.success("✅ Komentar ini **tidak mengandung cyberbullying.**")
        else:
            st.error(f"⚠️ Komentar ini termasuk **cyberbullying kategori: {label.upper()}**")

        # (Opsional) Tampilkan label untuk debugging
        # st.write("Label prediksi mentah:", label)
