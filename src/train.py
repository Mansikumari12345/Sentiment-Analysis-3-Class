"""
3-class sentiment analysis of Amazon product reviews:
negative / neutral / positive.

The dataset has star ratings from 1 to 5. I turn them into three classes:
    1-2 stars -> negative,  3 stars -> neutral,  4-5 stars -> positive

What this script does, step by step:
  1. Downloads the public dataset once and checks that the file is not corrupted.
  2. Builds a balanced sample (same number of reviews in each class).
  3. Splits it into a TRAIN part and a TEST part. TEST is not used until the end.
  4. Builds a pipeline: TF-IDF features -> classifier.
  5. Compares models and tunes them with GridSearchCV (5-fold cross-validation)
     on the TRAIN part only.
  6. Tests the best model once on the TEST part and saves metrics,
     a confusion matrix image and the trained model.

Run:  python src/train.py
Quick test (about 1 minute):  python src/train.py --per-class 3000
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw images without needing a screen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "amazon_reviews_en_train.parquet"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
# Hugging Face dataset "SetFit/amazon_reviews_multi_en": English Amazon reviews
# with star ratings (200,000 reviews in the train file, 40,000 per star rating).
DATA_URL = (
    "https://huggingface.co/api/datasets/SetFit/amazon_reviews_multi_en/"
    "parquet/default/train/0.parquet"
)
MIN_FILE_BYTES = 20_000_000  # the real file is about 25 MB
RANDOM_STATE = 42
CLASS_NAMES = ["negative", "neutral", "positive"]


def file_is_good():
    """True if the dataset file exists, has a sensible size and can be read."""
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size < MIN_FILE_BYTES:
        return False
    try:
        pd.read_parquet(DATA_FILE, columns=["label"])
        return True
    except Exception:
        return False


def download_data():
    """Download the dataset once. If the file is broken, download it again."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2, 3):
        if file_is_good():
            return
        if DATA_FILE.exists():
            print("Dataset file looks incomplete or broken. Deleting it.")
            DATA_FILE.unlink()
        print(f"Downloading dataset (attempt {attempt} of 3, about 25 MB) ...")
        try:
            request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                DATA_FILE.write_bytes(response.read())
        except Exception as error:
            print(f"Download failed: {error}")
    if not file_is_good():
        raise SystemExit("Could not download a valid dataset file. Check your internet and run again.")


def load_balanced_data(per_class):
    """Return texts and labels (0 negative, 1 neutral, 2 positive), equal size per class."""
    df = pd.read_parquet(DATA_FILE, columns=["text", "label"])
    stars = df["label"] + 1  # the file stores stars 1-5 as 0-4

    def sample(mask, n):
        return df[mask].sample(n=n, random_state=RANDOM_STATE)

    half = per_class // 2
    negative = pd.concat([sample(stars == 1, half), sample(stars == 2, per_class - half)])
    neutral = sample(stars == 3, per_class)
    positive = pd.concat([sample(stars == 4, half), sample(stars == 5, per_class - half)])
    negative = negative.assign(sentiment=0)
    neutral = neutral.assign(sentiment=1)
    positive = positive.assign(sentiment=2)
    data = pd.concat([negative, neutral, positive]).sample(frac=1, random_state=RANDOM_STATE)
    return data["text"].fillna(""), data["sentiment"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=30_000,
                        help="reviews per class (default 30000, so 90000 in total, max 40000)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="CPU cores for GridSearchCV (default 1, the safest on Windows)")
    args = parser.parse_args()

    # Create the output folders if they do not exist yet.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    download_data()
    X, y = load_balanced_data(args.per_class)
    print(f"Reviews used: {len(X)} ({args.per_class} per class)")

    # Hold out 20% as a final TEST set. stratify keeps the classes equal in size.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # K-Fold cross-validation (5 folds), keeping class balance in every fold.
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # TF-IDF turns text into numbers: words (and word pairs) weighted by how
    # informative they are. It is fitted inside the pipeline, so each CV fold
    # learns its vocabulary from its own training part only (no leakage).
    def make_pipeline(classifier):
        return Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=3,
                                      max_features=150_000)),
            ("clf", classifier),
        ])

    candidates = {
        "Naive Bayes": (
            make_pipeline(MultinomialNB()),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=300)),
            # Logistic Regression is the slowest model here, so I search a smaller grid.
            {"tfidf__ngram_range": [(1, 2)], "clf__C": [1, 4]},
        ),
        "Linear SVM": (
            make_pipeline(LinearSVC()),
            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 0.5, 1]},
        ),
    }

    searches = {}
    summary = []
    for name, (pipe, grid) in candidates.items():
        print(f"\nTuning {name} with GridSearchCV ({args.cv_folds}-fold) ...")
        start = time.time()
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=args.n_jobs)
        search.fit(X_train, y_train)
        searches[name] = search
        print(f"  best CV accuracy: {search.best_score_:.4f} | params: {search.best_params_}"
              f" | {time.time() - start:.0f}s")
        summary.append({
            "model": name,
            "best_cv_accuracy": round(search.best_score_, 4),
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
        })
        pd.DataFrame(search.cv_results_).to_csv(
            RESULTS_DIR / f"cv_results_{name.replace(' ', '_').lower()}.csv", index=False)

    # Pick the model with the best cross-validation score (not the test score!).
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best = searches[best_name].best_estimator_
    print(f"\nBest model by cross-validation: {best_name}")

    # Final, one-time evaluation on the untouched TEST set.
    pred = best.predict(X_test)
    report = classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(y_test, pred, target_names=CLASS_NAMES,
                                        digits=4, output_dict=True)
    metrics = {
        "dataset": "SetFit/amazon_reviews_multi_en (train file), "
                   "1-2 stars negative, 3 stars neutral, 4-5 stars positive",
        "reviews_per_class": args.per_class,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_folds": args.cv_folds,
        "best_model": best_name,
        "best_params": {k: str(v) for k, v in searches[best_name].best_params_.items()},
        "cv_accuracy": round(searches[best_name].best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "per_class": {c: {k: round(report_dict[c][k], 4) for k in ("precision", "recall", "f1-score")}
                      for c in CLASS_NAMES},
        "all_models_cv": summary,
    }
    print("\nTEST results:")
    print(report)

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "classification_report.txt").write_text(report)

    # Confusion matrix image (rows = true class, columns = predicted class).
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=CLASS_NAMES, cmap="Blues", ax=ax)
    ax.set_title(f"{best_name}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)

    # Save the trained model so predict.py can use it.
    joblib.dump(best, MODELS_DIR / "sentiment_3class_model.joblib")
    print("\nSaved: results/metrics.json, results/confusion_matrix.png, "
          "models/sentiment_3class_model.joblib")


if __name__ == "__main__":
    main()
