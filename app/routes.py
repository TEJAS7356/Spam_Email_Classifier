from pathlib import Path

import joblib
from flask import Blueprint, current_app, flash, render_template, request

main = Blueprint("main", __name__)


def _load_artifacts():
    """Load the vectorizer and classifier once, on the first request."""
    if not hasattr(current_app, "ml_model"):
        model_dir = Path(current_app.config["MODEL_DIR"])
        model_path = model_dir / "spam_classifier.joblib"
        vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                "Model files are missing. Run `python train_model.py` first."
            )
        current_app.ml_model = joblib.load(model_path)
        current_app.tfidf_vectorizer = joblib.load(vectorizer_path)
    return current_app.ml_model, current_app.tfidf_vectorizer


@main.route("/", methods=["GET", "POST"])
def index():
    subject = ""
    message = ""
    result = None

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        # The trained model classifies the complete email context together.
        email_text = f"Subject: {subject}\nMessage: {message}".strip()

        if not subject and not message:
            flash("Please enter a subject, message, or both.", "error")
        else:
            try:
                model, vectorizer = _load_artifacts()
                features = vectorizer.transform([email_text])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                class_probabilities = dict(zip(model.classes_, probabilities))
                spam_probability = float(class_probabilities.get("spam", 0.0))
                result = {
                    "label": "SPAM" if prediction == "spam" else "NOT SPAM",
                    "is_spam": prediction == "spam",
                    "confidence": max(probabilities) * 100,
                    "spam_probability": spam_probability * 100,
                }
            except FileNotFoundError as error:
                flash(str(error), "error")

    return render_template("index.html", subject=subject, message=message, result=result)
