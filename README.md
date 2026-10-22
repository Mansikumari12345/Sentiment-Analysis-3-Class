# 3-Class Sentiment Analysis of Amazon Product Reviews

I built a model that reads an Amazon product review and predicts whether it is **negative**, **neutral** or **positive**. It uses TF-IDF features and classic machine learning models from Scikit-Learn. I tuned three models with GridSearchCV and 5-fold cross-validation, then tested the best one once on 18,000 reviews it had never seen. I also built a small local web demo (Streamlit) where I can type a review and see the label and the scores.

**Author:** Mansi | [GitHub](https://github.com/Mansikumari12345) | [LinkedIn](https://www.linkedin.com/in/mansi-singh-168a05389) | mansi2003ssm@gmail.com

## How I made three classes

The dataset has star ratings from 1 to 5. I turned them into three classes: **1-2 stars = negative, 3 stars = neutral, 4-5 stars = positive**. I took the same number of reviews from each class (30,000 each, 90,000 in total), so the classes are balanced and accuracy is a fair metric.

## Results

Results of my run (80% train / 20% test, 72,000 train and 18,000 test reviews):

| Model (best settings found by 5-fold CV) | CV accuracy |
|---|---|
| Naive Bayes | 0.6683 |
| **Logistic Regression** (C=1, 1-2 word n-grams) | **0.6792** |
| Linear SVM (C=0.1, 1-2 word n-grams) | 0.6783 |

Logistic Regression had the best cross-validation score, so I tested it once on the test set:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.6966 | 0.7015 | 0.6991 |
| neutral | 0.5647 | 0.5732 | 0.5689 |
| positive | 0.7856 | 0.7683 | 0.7769 |

Test accuracy: **0.6810**. Macro F1: **0.6816**. (Guessing at random would give about 0.33.)

Confusion matrix: [results/confusion_matrix.png](results/confusion_matrix.png). Full report: [results/classification_report.txt](results/classification_report.txt). All numbers: [results/metrics.json](results/metrics.json).

**Neutral is the hardest class.** A 3-star review often mixes good and bad points, and the model often confuses it with negative or positive (see the confusion matrix). Even people disagree on whether a 3-star review is neutral, so I treat about 68% as a reasonable baseline, not a finished product.

## Dataset

[SetFit/amazon_reviews_multi_en](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) on Hugging Face (English Amazon product reviews with 1-5 star ratings). My script downloads the train file (about 25 MB), checks that it is not corrupted, and downloads it again if it is.

## How it works

1. **Balanced sample:** the same number of reviews per class.
2. **Split:** 20% of the reviews are held out as a test set that I do not touch until the end.
3. **Features:** TF-IDF on words and word pairs, up to 150,000 features.
4. **Models:** Naive Bayes, Logistic Regression and Linear SVM.
5. **Tuning:** GridSearchCV with stratified 5-fold cross-validation on the training part only. TF-IDF is fitted inside each fold, so nothing leaks from validation into training.
6. **Selection:** I pick the best model by cross-validation score, then evaluate it once on the test set.

## Run it (Windows, Command Prompt)

```
cd Sentiment-3-Class-Amazon-Reviews
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\train.py
```

Training takes about 10 to 20 minutes on one CPU core. For a quick test (about 1 minute): `python src\train.py --per-class 3000`.
The script creates the `results` and `models` folders itself and downloads the dataset itself.

Try one review in the terminal:

```
python src\predict.py "It's okay, nothing special"
```

## Web demo (runs only on my computer)

After training, start the demo:

```
streamlit run app.py
```

A browser tab opens at http://localhost:8501. I type a review (or click an example button), press Predict, and see the label and a percentage for each class. Press Ctrl+C in the terminal to stop it. The demo is not hosted online.

## Project structure

```
app.py            local web demo (Streamlit)
src/train.py      full pipeline: download, balance, split, tune, evaluate, save
src/predict.py    try the trained model on your own text in the terminal
results/          metrics, confusion matrix, cross-validation tables
requirements.txt  Python packages
```

## Limitations

- Neutral (3-star) reviews are the hardest class: precision and recall for neutral are about 0.57, much lower than for positive.
- The star rating is only a rough label of sentiment. Some 3-star reviews are clearly positive or negative in words, and some 2-star and 4-star reviews sound neutral.
- TF-IDF counts words and does not understand context, sarcasm or negation well. Transformer models such as BERT usually do better on this task. I have not tried them here.
- I used a balanced sample of 90,000 reviews so training finishes in reasonable time on a laptop. More data could help a little.
- The best C for Logistic Regression (1) is at the edge of the range I searched, so a wider search might change the result slightly.
- The probabilities in the demo come from Logistic Regression and can be over-confident. They are not guaranteed to be calibrated.
- One random sample and one split (seed 42). Another sample would give slightly different numbers.

## Future work

- Compare with a transformer model such as DistilBERT.
- Use more reviews and look at the misclassified neutral reviews to find patterns.
- Calibrate the probabilities.
