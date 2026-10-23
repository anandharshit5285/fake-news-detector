Fake News Detector
Live Demo
https://fake-news-detector-4rxu.onrender.com/
# 🛡️ Fake News Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?style=for-the-badge&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

An **AI-powered web application** that detects whether a news article is **Real** or **Fake**  
using Machine Learning and Natural Language Processing (NLP).

[🚀 Live Demo](#) • [📖 Documentation](#️-installation) • [🐛 Report Bug](https://github.com/sohailkhan-code/fake-news-detection/issues) • [💡 Request Feature](https://github.com/sohailkhan-code/fake-news-detection/issues)

</div>

---

## 📌 Table of Contents

- [About The Project](#-about-the-project)
- [Features](#-features)
- [Technologies Used](#️-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [ML Workflow](#-machine-learning-workflow)
- [Model Performance](#-model-performance)
- [Dataset Info](#-dataset-information)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 About The Project

Misinformation and fake news spread **6x faster** than real news on social media, affecting public opinion, elections, and everyday decisions. This project aims to combat that using **Artificial Intelligence**.

The system analyzes news articles and classifies them as **Real** or **Fake** by learning linguistic patterns, writing styles, and keyword usage from a dataset of **20,000+ labeled news articles**.

> 💡 **Key Highlight:** Complete end-to-end ML pipeline — from raw data to deployed web application.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🔍 Real-time Detection | Instant analysis of any news article |
| 📊 Confidence Score | Shows prediction confidence percentage |
| 🧪 Detailed Analysis | Top keywords, text statistics, cleaned text |
| 💻 Responsive UI | Works on desktop and mobile devices |
| ⚡ Fast Predictions | Results in under 1 second |
| 📈 Multiple Models | Tested 5 algorithms, best one deployed |
| 🔄 REST API | `/predict` and `/analyze` endpoints available |

---

## 🛠️ Technologies Used

### Frontend
| Technology | Purpose |
|------------|---------|
| HTML5 | Page structure |
| CSS3 | Styling & animations |
| JavaScript (Vanilla) | Dynamic interactions & API calls |

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| Flask 2.0+ | Web framework & REST API |
| Joblib | Model serialization |

### Machine Learning & NLP
| Library | Version | Purpose |
|---------|---------|---------|
| Scikit-learn | 1.0+ | ML models & TF-IDF |
| NLTK | 3.7+ | Text preprocessing |
| Pandas | 1.4+ | Data manipulation |
| NumPy | 1.22+ | Numerical operations |
| Matplotlib | 3.5+ | Visualizations |
| Seaborn | 0.11+ | Statistical plots |

---

## 📂 Project Structure

```text
fake-news-detection/
│
├── 📁 data/
│   ├── news.csv                    # Original Kaggle dataset
│   └── news_clean.csv              # Cleaned dataset
│
├── 📁 models/
│   ├── fake_news_model.pkl         # Trained ML model
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   ├── confusion_matrix.png        # Confusion matrix plot
│   ├── model_comparison.png        # Model comparison chart
│   ├── roc_curve.png               # ROC curve plot
│   └── eda_text_stats.png          # EDA visualizations
│
├── 📁 notebooks/
│   └── eda_and_training.ipynb      # Jupyter notebook (EDA + training)
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── style.css               # Custom styles
│   └── 📁 js/
│       └── script.js               # Frontend logic
│
├── 📁 templates/
│   └── index.html                  # Main HTML page
│
├── app.py                          # Flask backend server
├── model_training.py               # ML training pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚙️ Installation

### Prerequisites

Make sure you have installed:
- Python 3.8+
- pip
- Git

---

### Step 1: Clone Repository

```bash
git clone https://github.com/sohailkhan-code/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Add Dataset

Download dataset from Kaggle and place it in `data/` folder:
- 📥 [Fake and Real News Dataset - Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

```bash
data/
└── news.csv   ← Place here
```

### Step 6: Train Model

```bash
python model_training.py
```

### Step 7: Run Application

```bash
python app.py
```

### Step 8: Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧠 Machine Learning Workflow

```
Raw Data (news.csv)
      ↓
Data Cleaning (remove nulls, duplicates)
      ↓
Text Preprocessing (lowercase, remove URLs, punctuation, numbers)
      ↓
Stopword Removal + Stemming (NLTK)
      ↓
TF-IDF Vectorization (5000 features, bigrams)
      ↓
Model Training (5 algorithms tested)
      ↓
Evaluation (Accuracy, F1, ROC-AUC)
      ↓
Best Model Selected → Saved as .pkl
      ↓
Flask API → Web Application
```

---

## 📊 Model Performance

### 🏆 All Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | CV Score |
|-------|----------|-----------|--------|----------|----------|
| **Logistic Regression** ✅ | **95.2%** | **94.8%** | **95.6%** | **95.2%** | **94.9%** |
| SVM (Linear) | 94.7% | 94.3% | 95.1% | 94.7% | 94.5% |
| Random Forest | 92.3% | 91.8% | 93.0% | 92.4% | 91.9% |
| Naive Bayes | 89.1% | 88.5% | 89.8% | 89.1% | 88.7% |
| Gradient Boosting | 91.5% | 91.0% | 92.1% | 91.5% | 91.1% |

> ✅ **Best Model: Logistic Regression** — Selected based on highest accuracy and F1-Score

---

### 📈 Best Model - Detailed Metrics

| Metric | REAL News | FAKE News | Overall |
|--------|-----------|-----------|---------|
| Precision | 95.6% | 94.8% | 95.2% |
| Recall | 94.4% | 96.1% | 95.2% |
| F1-Score | 95.0% | 95.4% | 95.2% |
| Support | ~4,000 | ~4,000 | ~8,000 |

---

### 🔧 Feature Extraction Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| Vectorizer | TF-IDF | Better than Bag-of-Words for text |
| Max Features | 5,000 | Top important words |
| N-gram Range | (1, 2) | Unigrams + Bigrams |
| Min Document Freq | 5 | Remove rare words |
| Max Document Freq | 70% | Remove too-common words |
| Sublinear TF | True | Reduces impact of high freq words |

---

## 📁 Dataset Information

| Property | Details |
|----------|---------|
| Source | Kaggle - Fake and Real News Dataset |
| Total Articles | ~20,000+ |
| Real News | ~10,000 articles |
| Fake News | ~10,000 articles |
| Language | English |
| Format | CSV |
| Text Column | `text` |
| Label Column | `label` (0 = Real, 1 = Fake) |

---

## 📸 Screenshots

> 📌 Add your screenshots here after taking them from the application.

| Home Page | Real News Result |
|-----------|-----------------|
| ![Home]<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/10fc4c37-b413-4f44-91b5-e1bc95dda983" />)
 | ![Real]<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b8d52436-5a88-485d-bac5-b61fb387857a" />) |

| Fake News Result | Detailed Analysis |
|-----------------|-------------------|
| ![Fake]<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b75803f4-0fc2-4c9a-95bb-8609f04dcc26" />)
| ![Analysis]<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7d15bccb-f2de-4d2c-8439-308740309cce" />) |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | POST | Quick prediction (Real/Fake) |
| `/analyze` | POST | Detailed analysis with keywords |
| `/health` | GET | Server health check |

### Example API Usage

```python
import requests

response = requests.post('http://127.0.0.1:5000/predict', 
    json={"text": "Your news article here..."})

print(response.json())
# Output:
# {
#   "prediction": "REAL",
#   "confidence": 87.3,
#   "label": 0,
#   "word_count": 45
# }
```

---

## 🔮 Future Improvements

| Feature | Priority | Status |
|---------|----------|--------|
| BERT / Transformer Models | High | 🔜 Planned |
| News Source Verification | High | 🔜 Planned |
| Fact-Checking API Integration | Medium | 💡 Idea |
| Multi-language Support | Medium | 💡 Idea |
| Chrome Browser Extension | High | 🔜 Planned |
| Mobile App (Flutter) | Low | 💡 Idea |
| User Authentication | Low | 💡 Idea |
| Live News API | Medium | 💡 Idea |

---

## 👨‍💻 Author

<div align="center">







</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If you found this project helpful:

- ⭐ **Star this repository**
- 🍴 **Fork and contribute**
- 🐛 **Report bugs via Issues**
- 📢 **Share with others**

---

<div align="center">

Made by **Harshit Anand  (2300970130056) ,Khushi Saraswat (2300970130069),Kanhaiya Sahani(2300970130067)**



</div>
