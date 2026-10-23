import pandas as pd, joblib, numpy as np, re, string, sys
from sklearn.metrics import accuracy_score

DATA_PATH = 'data/news.csv'
MODEL_PATH = 'models/fake_news_model.pkl'
VEC_PATH = 'models/tfidf_vectorizer.pkl'

# Load dataset
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
except Exception as e:
    print('Error loading dataset:', e)
    sys.exit(1)

# Identify relevant columns
title_col = None
text_col = None
label_col = None
for col in df.columns:
    low = col.lower()
    if low in ['title', 'headline'] and title_col is None:
        title_col = col
    if low in ['text', 'content', 'article', 'news'] and text_col is None:
        text_col = col
    if low in ['label', 'class', 'target', 'is_fake'] and label_col is None:
        label_col = col

print('Identified columns -> Title:', title_col, '| Text:', text_col, '| Label:', label_col)

# Distribution of labels
print('\n--- Label distribution ---')
print(df[label_col].value_counts())

# Verify correct mapping (REAL=0, FAKE=1)
expected = {'REAL', 'FAKE'}
actual_labels = set(df[label_col].unique())
if actual_labels != expected:
    print('\n[WARNING] Unexpected label values:', actual_labels)
else:
    print('\nLabel values as expected (REAL/FAKE).')

# Create combined field (title + text) for consistency with training
df['combined'] = df[title_col].astype(str) + ' ' + df[text_col].astype(str)

# Sample 10 REAL and 10 FAKE headlines (using title only for readability)
real_titles = df[df[label_col] == 'REAL'][title_col].dropna().sample(10, random_state=42).tolist()
fake_titles = df[df[label_col] == 'FAKE'][title_col].dropna().sample(10, random_state=42).tolist()

samples = real_titles + fake_titles
actual = ['REAL'] * 10 + ['FAKE'] * 10

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# Preprocessing (same as app.py)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def preprocess(text):
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

preprocessed = [preprocess(t) for t in samples]
X = vectorizer.transform(preprocessed)

# Predictions
pred_int = model.predict(X)
int_to_label = {0: 'REAL', 1: 'FAKE'}
pred_labels = [int_to_label[i] for i in pred_int]

# Confidence scores
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)
    confidences = np.max(probs, axis=1) * 100
else:
    # LinearSVC provides decision_function; map to probability via sigmoid
    decision = model.decision_function(X)
    probs = 1 / (1 + np.exp(-decision))
    confidences = probs * 100

print('\n--- Sample predictions (headline, actual, predicted, confidence) ---')
for h, act, pred, conf in zip(samples, actual, pred_labels, confidences):
    print(f'Headline: {h}\n  Actual: {act} | Predicted: {pred} | Confidence: {conf:.2f}%\n')
