import pandas as pd, numpy as np, re, string, joblib, nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------- Preprocessing (identical to app.py) ----------
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# ---------- Load dataset ----------
DATA_PATH = 'data/news.csv'
MODEL_PATH = 'models/fake_news_model.pkl'  # filenames used by Flask app
VEC_PATH = 'models/tfidf_vectorizer.pkl'

print('Loading dataset...')
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')

# Identify required columns
title_col = text_col = label_col = None
for col in df.columns:
    low = col.lower()
    if low in ['title', 'headline'] and not title_col:
        title_col = col
    if low in ['text', 'content', 'article', 'news'] and not text_col:
        text_col = col
    if low in ['label', 'class', 'target', 'is_fake'] and not label_col:
        label_col = col

if not all([title_col, text_col, label_col]):
    raise ValueError('Required columns (title, text, label) not found')

# Combine title and text, then apply preprocessing (must match inference)
df['combined_raw'] = df[title_col].astype(str) + ' ' + df[text_col].astype(str)
df['combined'] = df['combined_raw'].apply(preprocess)

# Clean data: drop missing / duplicate rows
initial_len = len(df)
df = df.dropna(subset=['combined', label_col])
df = df.drop_duplicates(subset=['combined', label_col])
print(f'Dropped {initial_len - len(df)} rows (empty or duplicate)')

# Deterministic label mapping (REAL=0, FAKE=1)
unique_labels = set(df[label_col].unique())
if unique_labels != {'REAL', 'FAKE'}:
    raise ValueError(f'Unexpected label set: {unique_labels}')
label_to_int = {'REAL': 0, 'FAKE': 1}
int_to_label = {v: k for k, v in label_to_int.items()}
print('Label mapping:', label_to_int)

y = df[label_col].map(label_to_int)
X = df['combined']

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# TF‑IDF vectorizer (same for both models)
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_vec, y_train)
log_pred = log_reg.predict(X_test_vec)
log_f1 = f1_score(y_test, log_pred, pos_label=1)
log_metrics = (
    accuracy_score(y_test, log_pred),
    precision_score(y_test, log_pred, pos_label=1),
    recall_score(y_test, log_pred, pos_label=1),
    log_f1,
)
print('LogisticRegression – Acc:{:.4f} Prec:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(*log_metrics))

# Linear SVC
svc = LinearSVC()
svc.fit(X_train_vec, y_train)
svc_pred = svc.predict(X_test_vec)
svc_f1 = f1_score(y_test, svc_pred, pos_label=1)
svc_metrics = (
    accuracy_score(y_test, svc_pred),
    precision_score(y_test, svc_pred, pos_label=1),
    recall_score(y_test, svc_pred, pos_label=1),
    svc_f1,
)
print('LinearSVC – Acc:{:.4f} Prec:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(*svc_metrics))

# Choose best model by F1
if svc_f1 > log_f1:
    best_model = svc
    best_metrics = svc_metrics
    best_name = 'LinearSVC'
else:
    best_model = log_reg
    best_metrics = log_metrics
    best_name = 'LogisticRegression'

print('Selected model:', best_name)
print('Selected metrics – Acc:{:.4f} Prec:{:.4f} Rec:{:.4f} F1:{:.4f}'.format(*best_metrics))

# Save model and vectorizer (filenames expected by Flask)
joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VEC_PATH)
print('Saved model to', MODEL_PATH)
print('Saved vectorizer to', VEC_PATH)

# Verify on 5 REAL and 5 FAKE samples (using preprocessed combined)
real_samples = df[df[label_col] == 'REAL']['combined'].sample(5, random_state=42).tolist()
fake_samples = df[df[label_col] == 'FAKE']['combined'].sample(5, random_state=42).tolist()
sample_texts = real_samples + fake_samples
actual = ['REAL'] * 5 + ['FAKE'] * 5
sample_vec = vectorizer.transform(sample_texts)
pred_int = best_model.predict(sample_vec)
pred_labels = [int_to_label[i] for i in pred_int]
print('\n--- Sample Predictions (post‑preprocess) ---')
for txt, act, pred in zip(sample_texts, actual, pred_labels):
    print(f'Actual: {act} | Predicted: {pred}')
