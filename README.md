# MailShield: Beginner-Friendly Spam Email Classifier

A complete Flask web application that classifies a pasted message as **Spam** or **Not Spam** using a real machine-learning pipeline: **TF-IDF Vectorization + Multinomial Naive Bayes**. It uses no keyword rules, external AI services, or prediction APIs.

## Machine-learning pipeline

```text
Email
  → Text preprocessing
  → TF-IDF vectorization
  → Multinomial Naive Bayes
  → Spam / Not Spam
```

The project trains on the public **SMS Spam Collection** dataset from the UCI Machine Learning Repository. Although the examples are SMS messages, the messages are short natural-language communications, so they are a practical educational dataset for demonstrating email/message spam classification.

## Project structure

```text
spam_email_classifier/
├── app/
│   ├── __init__.py          # Flask application factory
│   ├── routes.py            # Prediction route and model loading
│   ├── static/style.css     # Responsive styling
│   └── templates/index.html # User interface
├── data/                    # Dataset downloaded by the training script
├── model/                   # Saved joblib model and vectorizer
├── train_model.py           # Download, train, evaluate, and save artifacts
├── run.py                   # Local Flask entry point
├── requirements.txt
└── README.md
```

## Setup and installation

Python 3.9 or newer is recommended.

```bash
# 1. Enter the project directory
cd spam_email_classifier

# 2. Create and activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

## Train the model

Run this once before starting Flask:

```bash
python train_model.py
```

The script downloads the dataset into `data/` if it is not already there. It then:

1. Cleans the text by lowercasing, replacing URLs and numbers with readable placeholders, and removing noisy punctuation.
2. Splits the labeled examples into 80% training data and 20% test data using a fixed random seed.
3. Fits a TF-IDF vectorizer on the training messages only.
4. Trains a `MultinomialNB` classifier.
5. Prints accuracy, precision, recall, F1-score, a classification report, and the confusion matrix.
6. Saves `model/spam_classifier.joblib` and `model/tfidf_vectorizer.joblib`.

## Run the Flask application

```bash
python run.py
```

Open <http://127.0.0.1:5000> in a browser. Enter the email **subject** and **message** in their separate fields, click **Classify email**, and the page will show the predicted class, confidence, and estimated spam probability. Flask combines both fields before sending the text through the trained model. **Clear** resets the form.

## Concepts in simple language

### What does TF-IDF do?

Text must become numbers before a machine-learning model can use it. TF-IDF gives a word a larger value when it is important in one message but not common across every message. Common filler words get less influence, while distinctive terms get more influence. This implementation also includes word pairs (bigrams), which can capture phrases such as “claim prize”.

### What does Naive Bayes do?

Naive Bayes learns how strongly words are associated with each class. During training it estimates the likelihood of seeing the words in spam messages and in non-spam messages. For a new message, it combines those learned probabilities and chooses the more likely class.

### Why is Multinomial Naive Bayes suitable for text?

After TF-IDF, a message is represented as a vector of non-negative word weights. Multinomial Naive Bayes is designed for this kind of count/weight-based feature representation, is fast to train, works well as a text-classification baseline, and is easy to explain.

### How does training work?

The labeled dataset contains each message and its label (`spam` or `ham`, where `ham` means not spam). The training portion teaches the vectorizer its vocabulary and teaches the classifier the relationship between text features and labels. The untouched test portion measures how well the trained pipeline generalizes to messages it did not see during training.

### How is a new email classified?

The Flask route sends the new text through the already-fitted vectorizer, producing the same kind of TF-IDF vector used during training. The saved Multinomial Naive Bayes model then returns `spam` or `ham` and class probabilities. The interface converts those labels into the friendlier **SPAM** and **NOT SPAM** display text.

## Notes

This is an educational classifier, not a replacement for a production mail server's layered security system. A model trained on the SMS Spam Collection may not represent every kind of modern email, and confidence is a model probability—not a guarantee.
