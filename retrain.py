import os
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# Helper functions for preprocessing (mirrors app.py preprocessing)
# ------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean and normalize raw text.
    Steps:
    1. Lowercase
    2. Remove URLs, emails, HTML tags, mentions, hashtags
    3. Remove punctuation and numbers
    4. Remove stopwords (NLTK) and apply Porter stemming
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@[\w_]+", " ", text)
    text = re.sub(r"#[\w_]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Identify columns (case‑insensitive)
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get('title')
    text_col = cols.get('text')
    label_col = cols.get('label')
    if not (title_col and text_col and label_col):
        raise ValueError('Dataset must contain title, text, and label columns')
    # Combine title and text
    df['combined'] = df[title_col].astype(str) + " " + df[text_col].astype(str)
    df = df.dropna(subset=['combined', label_col])
    df['clean'] = df['combined'].apply(clean_text)
    df = df.drop_duplicates(subset=['clean'])
    # Map labels to numeric
    unique_labels = df[label_col].unique()
    if set(map(str.upper, unique_labels)) == {"REAL", "FAKE"}:
        label_map = {"REAL": 0, "FAKE": 1}
    else:
        # fallback numeric mapping
        label_map = {lbl: idx for idx, lbl in enumerate(sorted(unique_labels))}
    df['label_num'] = df[label_col].map(label_map)
    return df, label_map

def train_and_evaluate(df: pd.DataFrame):
    X = df['clean']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr.fit(X_train_vec, y_train)
    lr_preds = lr.predict(X_test_vec)
    lr_probs = lr.predict_proba(X_test_vec)[:, 1]
    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_preds),
        'precision': precision_score(y_test, lr_preds),
        'recall': recall_score(y_test, lr_preds),
        'f1': f1_score(y_test, lr_preds),
    }

    # LinearSVC
    svc = LinearSVC()
    svc.fit(X_train_vec, y_train)
    svc_preds = svc.predict(X_test_vec)
    svc_decision = svc.decision_function(X_test_vec)
    svc_probs = 1 / (1 + np.exp(-svc_decision))
    svc_metrics = {
        'accuracy': accuracy_score(y_test, svc_preds),
        'precision': precision_score(y_test, svc_preds),
        'recall': recall_score(y_test, svc_preds),
        'f1': f1_score(y_test, svc_preds),
    }

    # Choose best by F1
    if lr_metrics['f1'] >= svc_metrics['f1']:
        best_model, best_name, best_metrics, best_probs = lr, 'LogisticRegression', lr_metrics, lr_probs
    else:
        best_model, best_name, best_metrics, best_probs = svc, 'LinearSVC', svc_metrics, svc_probs

    return best_model, best_name, best_metrics, vectorizer, best_probs, X_test, y_test

def save_artifacts(model, vectorizer, model_path, vec_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Saved model to {model_path}\nSaved vectorizer to {vec_path}")

def evaluate_samples(model, vectorizer, label_map, csv_path='data/news.csv'):
    """Select 10 REAL and 10 FAKE headlines, predict, and report per‑group accuracy."""
    inv_map = {v: k for k, v in label_map.items()}
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get('title')
    text_col = cols.get('text')
    label_col = cols.get('label')
    df['combined'] = df[title_col].astype(str) + " " + df[text_col].astype(str)
    df = df.dropna(subset=['combined', label_col])
    df['clean'] = df['combined'].apply(clean_text)
    # Get 10 of each label (case‑insensitive)
    real_df = df[df[label_col].str.upper() == 'REAL'].head(10)
    fake_df = df[df[label_col].str.upper() == 'FAKE'].head(10)
    sample_df = pd.concat([real_df, fake_df])
    X = vectorizer.transform(sample_df['clean'])
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = 1 / (1 + np.exp(-model.decision_function(X)))
    preds = model.predict(X)
    # Print each sample
    print('\nSample predictions (headline, actual, predicted, confidence):')
    for i, row in enumerate(sample_df.itertuples()):
        headline = getattr(row, title_col)
        actual = getattr(row, label_col).upper()
        pred = inv_map[preds[i]]
        confidence = probs[i]
        print(f"- {headline}\n  Actual: {actual}, Predicted: {pred}, Confidence: {confidence:.3f}\n")
    # Compute per‑group accuracy
    real_mask = sample_df[label_col].str.upper() == 'REAL'
    fake_mask = sample_df[label_col].str.upper() == 'FAKE'
    real_acc = np.mean(preds[real_mask.values] == 0)  # label 0 = REAL
    fake_acc = np.mean(preds[fake_mask.values] == 1)  # label 1 = FAKE
    print(f"Real-sample accuracy (10 samples): {real_acc:.2%}")
    print(f"Fake-sample accuracy (10 samples): {fake_acc:.2%}")

if __name__ == "__main__":
    csv_path = os.path.join('data', 'news.csv')
    model_path = os.path.join('models', 'fake_news_model.pkl')
    vec_path = os.path.join('models', 'tfidf_vectorizer.pkl')

    print("Loading dataset...")
    df, label_map = load_dataset(csv_path)
    print(f"Dataset size after cleaning: {len(df)} rows")
    print("Training models...")
    best_model, best_name, best_metrics, vectorizer, _, _, _ = train_and_evaluate(df)
    print(f"Best model: {best_name}")
    print("Metrics on held-out test set:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    save_artifacts(best_model, vectorizer, model_path, vec_path)
    evaluate_samples(best_model, vectorizer, label_map, csv_path)
