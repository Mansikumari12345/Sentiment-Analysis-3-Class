"""Try the trained 3-class model on your own review text.

Run train.py first, then:
    python src/predict.py "This phone is great, the battery lasts all day"
    python src/predict.py "It's okay, nothing special"
    python src/predict.py            (asks you to type a review)

It prints the label (negative / neutral / positive) and a score for each class.
"""
import math
import sys
from pathlib import Path

import joblib

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "sentiment_3class_model.joblib"
CLASS_NAMES = ["negative", "neutral", "positive"]

if not MODEL_FILE.exists():
    raise SystemExit("Model not found. Run  python src/train.py  first.")

model = joblib.load(MODEL_FILE)
text = " ".join(sys.argv[1:]) or input("Type a review: ")

if hasattr(model, "predict_proba"):
    # Logistic Regression and Naive Bayes give real probabilities.
    scores = model.predict_proba([text])[0]
    kind = "probability"
else:
    # Linear SVM gives distance scores. I turn them into percentages with
    # softmax so they add up to 100%. These are NOT true probabilities.
    raw = model.decision_function([text])[0]
    exps = [math.exp(v) for v in raw]
    scores = [v / sum(exps) for v in exps]
    kind = "confidence (from SVM scores, not a true probability)"

best = max(range(len(scores)), key=lambda i: scores[i])
print("Prediction:", CLASS_NAMES[best])
print(f"Score type: {kind}")
for name, value in zip(CLASS_NAMES, scores):
    print(f"  {name:<9} {value * 100:5.1f}%")
