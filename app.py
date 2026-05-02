import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------
# Load Model & Vectorizer (Safe Loading)
# ---------------------------
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(words)

# ---------------------------
# UI Configuration
# ---------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.markdown("### Detect whether a news article is **Fake** or **Real**")

st.info("This model analyzes writing patterns and may not verify factual truth.")

st.divider()

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Enter News Text")

user_input = st.text_area(
    "Paste your news content here:",
    height=200,
    placeholder="Type or paste news content..."
)

st.subheader("Or Upload File")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

st.divider()

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Analyze News"):

    # Input Handling
    if user_input.strip():
        text = user_input

    elif uploaded_file is not None:
        try:
            text = uploaded_file.read().decode("utf-8")
        except:
            st.error("Error reading file. Please upload a valid .txt file.")
            st.stop()

    else:
        st.warning("⚠ Please enter text or upload a file")
        st.stop()

    # Preprocess
    cleaned = clean_text(text)

    if len(cleaned.strip()) == 0:
        st.warning("⚠ Input text is too short after cleaning.")
        st.stop()

    # Vectorize
    vectorized = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    fake_prob = probability[0]
    real_prob = probability[1]

    # ---------------------------
    # Output Section
    # ---------------------------
    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.success("✅ This news is likely REAL")
        confidence = real_prob
    else:
        st.error("❌ This news is likely FAKE")
        confidence = fake_prob

    # Confidence
    st.write(f"### Confidence Score: {confidence:.2f}")
    st.progress(float(confidence))

    # Probability Breakdown
    st.markdown("### Probability Breakdown")
    st.write(f"🔴 Fake Probability: **{fake_prob:.2f}**")
    st.write(f"🟢 Real Probability: **{real_prob:.2f}**")

    # Optional Expand Section
    with st.expander("See Processed Text"):
        st.write(cleaned)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Developed as part of a Machine Learning Course Project")