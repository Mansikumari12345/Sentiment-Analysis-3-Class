"""Small local demo: type a review, see the predicted sentiment and the scores.

Run (from the project folder):
    streamlit run app.py
It opens in your browser at http://localhost:8501 and runs only on your computer.
Run  python src/train.py  once first, so the trained model exists.
"""
import math
from pathlib import Path

import joblib
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "models" / "sentiment_3class_model.joblib"
CLASS_NAMES = ["negative", "neutral", "positive"]
EXAMPLES = {
    "Great product": "Great product, works perfectly and arrived early. Totally worth the price.",
    "It's okay": "It's okay, nothing special. Does the job but I expected a bit more.",
    "Not bad, not great": "Not bad but not great either. Average quality for the price.",
    "Terrible": "Terrible quality. It broke after two days and support never replied.",
}

st.set_page_config(page_title="Review Sentiment Demo", page_icon=None)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def score(model, text):
    """Return one score per class. Probabilities if the model has them,
    otherwise softmax of the SVM scores (not true probabilities)."""
    if hasattr(model, "predict_proba"):
        return list(model.predict_proba([text])[0]), "Probabilities"
    raw = model.decision_function([text])[0]
    exps = [math.exp(v) for v in raw]
    return [v / sum(exps) for v in exps], "Confidence (from SVM scores, not true probabilities)"


st.title("Amazon Review Sentiment Demo")
st.write("Type a product review. The model predicts **negative**, **neutral** or **positive**.")

if not MODEL_FILE.exists():
    st.error("Model not found. Open a terminal in this folder and run:  python src/train.py")
    st.stop()

model = load_model()

if "review" not in st.session_state:
    st.session_state["review"] = ""

st.write("Try an example:")
columns = st.columns(len(EXAMPLES))
for column, (label, example) in zip(columns, EXAMPLES.items()):
    if column.button(label):
        st.session_state["review"] = example

review = st.text_area("Your review", key="review", height=120,
                      placeholder="Type or paste a review here...")

if st.button("Predict", type="primary"):
    if not review.strip():
        st.warning("Please type a review first.")
    else:
        values, kind = score(model, review)
        best = max(range(len(values)), key=lambda i: values[i])
        st.subheader(f"Prediction: {CLASS_NAMES[best]}")
        st.caption(kind)
        for name, value in zip(CLASS_NAMES, values):
            st.write(f"{name.capitalize()}: {value * 100:.1f}%")
            st.progress(float(value))

st.divider()
st.caption("Built by Mansi. Model: TF-IDF + a classic scikit-learn classifier trained on "
           "Amazon reviews (1-2 stars negative, 3 neutral, 4-5 positive). "
           "Neutral is the hardest class, so expect mistakes on mixed reviews.")
