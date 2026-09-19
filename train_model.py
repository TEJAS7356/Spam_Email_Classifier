"""Train and evaluate the spam classifier.

Dataset: SMS Spam Collection, a public collection of labeled SMS messages.
The same text-classification workflow applies to short email messages.
"""
from pathlib import Path
import re
import urllib.request

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = DATA_DIR / "SMSSpamCollection"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def download_dataset():
    """Download and extract the dataset only when it is not already present."""
    if DATA_PATH.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "smsspamcollection.zip"
    print("Downloading the SMS Spam Collection dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DATA_DIR)
    zip_path.unlink(missing_ok=True)


def clean_text(text):
    """Simple, explainable preprocessing: lowercase and normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z_ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    download_dataset()
    data = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    data["clean_message"] = data["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_message"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    # Fit TF-IDF only on training text to avoid leaking test-set information.
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    predictions = classifier.predict(X_test_tfidf)

    print("\nEvaluation on the held-out test set")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='spam'):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, pos_label='spam'):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, pos_label='spam'):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["not spam", "spam"]))
    print("Confusion matrix (rows = actual, columns = predicted; order: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(classifier, MODEL_DIR / "spam_classifier.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    print(f"\nSaved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
