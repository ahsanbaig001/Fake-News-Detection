import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Download required NLTK data (only runs once)
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Paste a news snippet below to check if it is **Fake** or **Real**.")

user_input = st.text_area("Enter news snippet:")

if st.button("Predict"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ This looks like **Real News**")
        else:
            st.error("❌ This looks like **Fake News**")
    else:
        st.warning("⚠️ Please enter some text before predicting.")
