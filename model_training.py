"""
Fake News Detection - Complete Model Training Pipeline
Ye file main model training, preprocessing, aur evaluation handle karegi
"""

import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
import warnings
warnings.filterwarnings('ignore')

# NLTK downloads
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class FakeNewsDetector:
    """
    Complete Fake News Detection System with advanced NLP preprocessing
    """
    
    def __init__(self, dataset_path='data/news.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tfidf_vectorizer = None
        self.best_model = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Dataset load karna with error handling"""
        print("\n" + "="*60)
        print("STEP 1: Loading Dataset...")
        print("="*60)
        
        try:
            # Try different methods to load CSV
            print("Attempting to load CSV file...")
            
            # Method 1: Standard read with error handling
            try:
                self.df = pd.read_csv(self.dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 encoding failed, trying latin-1...")
                self.df = pd.read_csv(self.dataset_path, encoding='latin-1')
            except pd.errors.ParserError:
                print("Standard parsing failed, trying with error handling...")
                self.df = pd.read_csv(self.dataset_path, 
                                     encoding='utf-8',
                                     on_bad_lines='skip',
                                     engine='python')
            
            print(f"✓ Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"\nColumns: {list(self.df.columns)}")
            print(f"\nFirst few rows:")
            print(self.df.head())
            
            # Check if dataset is empty
            if len(self.df) == 0:
                raise ValueError("Dataset is empty!")
            
            return self.df
            
        except FileNotFoundError:
            print("\n❌ ERROR: Dataset file not found!")
            print(f"Looking for: {self.dataset_path}")
            print("\n📌 Solution:")
            print("1. Download dataset from Kaggle")
            print("2. Place it in 'data/' folder")
            print("3. Rename it to 'news.csv'")
            print("\nPopular datasets:")
            print("- https://www.kaggle.com/c/fake-news/data")
            print("- https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
            raise
        except Exception as e:
            print(f"\n❌ ERROR loading dataset: {str(e)}")
            print("\n📌 Possible solutions:")
            print("1. Check if CSV file is corrupted")
            print("2. Try opening CSV in Excel and re-saving")
            print("3. Make sure file has proper headers")
            raise
    
    def exploratory_data_analysis(self):
        """Complete EDA - Data ko samajhna"""
        print("\n" + "="*60)
        print("STEP 2: Exploratory Data Analysis (EDA)")
        print("="*60)
        
        # Basic info
        print("\n--- Dataset Info ---")
        print(self.df.info())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        print("\n--- Missing Values ---")
        print(self.df.isnull().sum())
        
        print("\n--- Duplicate Rows ---")
        duplicates = self.df.duplicated().sum()
        print(f"Total duplicates: {duplicates}")
        
        # Label distribution (assume column name 'label' ya 'class')
        label_col = None
        for col in ['label', 'class', 'target', 'is_fake']:
            if col in self.df.columns:
                label_col = col
                break
        
        if label_col:
            print(f"\n--- {label_col.upper()} Distribution ---")
            print(self.df[label_col].value_counts())
            
            # Visualization
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            self.df[label_col].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
            plt.title('Fake vs Real News Distribution')
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            
            plt.subplot(1, 2, 2)
            self.df[label_col].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                                     colors=['#FF6B6B', '#4ECDC4'])
            plt.title('Label Distribution %')
            plt.ylabel('')
            
            plt.tight_layout()
            plt.savefig('models/eda_label_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Label distribution plot saved!")
            plt.close()
        
        # Text length analysis
        text_col = None
        for col in ['text', 'content', 'article', 'news']:
            if col in self.df.columns:
                text_col = col
                break
        
        if text_col:
            self.df['text_length'] = self.df[text_col].astype(str).apply(len)
            self.df['word_count'] = self.df[text_col].astype(str).apply(lambda x: len(x.split()))
            
            print(f"\n--- Text Statistics ---")
            print(f"Average text length: {self.df['text_length'].mean():.2f}")
            print(f"Average word count: {self.df['word_count'].mean():.2f}")
            
            # Text length visualization
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.df['text_length'], bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
            plt.title('Text Length Distribution')
            plt.xlabel('Length')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.hist(self.df['word_count'], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
            plt.title('Word Count Distribution')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('models/eda_text_stats.png', dpi=300, bbox_inches='tight')
            print("✓ Text statistics plot saved!")
            plt.close()
    
    def preprocess_text(self, text):
        """
        Advanced text preprocessing with multiple techniques
        Ye function har news article ko clean karega
        """
        # Convert to string
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = text.split()
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Stemming (optional - can use lemmatization instead)
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Lemmatization (more accurate but slower)
        # tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def prepare_data(self):
        """Data preparation and preprocessing"""
        print("\n" + "="*60)
        print("STEP 3: Data Preprocessing & Feature Engineering")
        print("="*60)
        
        # Find text and label columns
        text_col = None
        label_col = None
        
        for col in ['text', 'content', 'article', 'news', 'title']:
            if col in self.df.columns:
                text_col = col
                break
        
        for col in ['label', 'class', 'target', 'is_fake']:
            if col in self.df.columns:
                label_col = col
                break
        
        if not text_col or not label_col:
            print("❌ Error: Could not find text or label column!")
            print(f"Available columns: {list(self.df.columns)}")
            return
        
        print(f"✓ Using text column: '{text_col}'")
        print(f"✓ Using label column: '{label_col}'")
        
        # Remove missing values
        print("\nRemoving missing values...")
        initial_shape = self.df.shape[0]
        self.df = self.df.dropna(subset=[text_col, label_col])
        print(f"Removed {initial_shape - self.df.shape[0]} rows with missing values")
        
        # Remove duplicates
        print("\nRemoving duplicates...")
        initial_shape = self.df.shape[0]
        self.df = self.df.drop_duplicates(subset=[text_col])
        print(f"Removed {initial_shape - self.df.shape[0]} duplicate rows")
        
        # Text preprocessing
        print("\nPreprocessing text data...")
        print("This may take a few minutes for large datasets...")
        self.df['cleaned_text'] = self.df[text_col].apply(self.preprocess_text)
        print("✓ Text preprocessing completed!")
        
        # Show sample
        print("\n--- Sample Preprocessed Text ---")
        for i in range(min(3, len(self.df))):
            print(f"\nOriginal: {self.df[text_col].iloc[i][:200]}...")
            print(f"Cleaned: {self.df['cleaned_text'].iloc[i][:200]}...")
        
        # Prepare features and labels
        X = self.df['cleaned_text']
        y = self.df[label_col]
        
        # Convert labels to binary if needed
        if y.dtype == 'object':
            print(f"\nConverting labels to binary...")
            print(f"Unique labels: {y.unique()}")
            
            # Check number of unique labels
            unique_labels = y.unique()
            
            if len(unique_labels) == 2:
                # Simple binary mapping
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                y = y.map(label_mapping)
                print(f"Label mapping: {label_mapping}")
            
            elif len(unique_labels) > 2:
                # Multiple labels - convert to binary
                print(f"\n⚠️  Found {len(unique_labels)} labels: {list(unique_labels)}")
                print("Converting to binary classification...")
                
                # Option 1: Treat 'misleading' as fake
                label_mapping = {}
                for label in unique_labels:
                    if 'real' in label.lower() or 'true' in label.lower():
                        label_mapping[label] = 0  # Real news
                    else:
                        label_mapping[label] = 1  # Fake/Misleading news
                
                y = y.map(label_mapping)
                print(f"Label mapping: {label_mapping}")
                print(f"  - Real news: {label_mapping}")
                print(f"  - Fake/Misleading news: mapped to 1")
            
            else:
                raise ValueError(f"Only one label found: {unique_labels}")
        
        # Train-test split
        print("\nSplitting data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Train set size: {len(self.X_train)}")
        print(f"✓ Test set size: {len(self.X_test)}")
        print(f"✓ Train label distribution:\n{self.y_train.value_counts()}")
        print(f"✓ Test label distribution:\n{self.y_test.value_counts()}")
    
    def create_tfidf_features(self):
        """
        TF-IDF Vectorization - Sabse important step!
        Ye text ko numerical features mein convert karega
        """
        print("\n" + "="*60)
        print("STEP 4: TF-IDF Feature Extraction")
        print("="*60)
        
        print("\nCreating TF-IDF features...")
        print("Parameters:")
        print("- max_features: 5000 (top 5000 important words)")
        print("- ngram_range: (1,2) (unigrams and bigrams)")
        print("- min_df: 5 (word must appear in at least 5 documents)")
        print("- max_df: 0.7 (word must appear in max 70% documents)")
        
        # Advanced TF-IDF with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=5,  # minimum document frequency
            max_df=0.7,  # maximum document frequency
            sublinear_tf=True,  # use sublinear tf scaling
            use_idf=True
        )
        
        # Fit and transform
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
        
        print(f"\n✓ TF-IDF transformation completed!")
        print(f"✓ Training features shape: {X_train_tfidf.shape}")
        print(f"✓ Testing features shape: {X_test_tfidf.shape}")
        print(f"✓ Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        # Show top features
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"\n--- Sample TF-IDF Features ---")
        print(f"Top 20 features: {list(feature_names[:20])}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_models(self, X_train_tfidf, X_test_tfidf):
        """
        Multiple models ko train karenge aur compare karenge
        """
        print("\n" + "="*60)
        print("STEP 5: Model Training & Evaluation")
        print("="*60)
        
        # Define multiple models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                C=1.0, 
                solver='liblinear',
                random_state=42
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            
            # Train
            model.fit(X_train_tfidf, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_tfidf)
            y_pred_test = model.predict(X_test_tfidf)
            
            # Metrics
            train_acc = accuracy_score(self.y_train, y_pred_train)
            test_acc = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test)
            recall = recall_score(self.y_test, y_pred_test)
            f1 = f1_score(self.y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_tfidf, self.y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_test
            }
            
            print(f"\n--- {name} Results ---")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Cross-Val Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred_test)
            print(f"\nConfusion Matrix:")
            print(cm)
            
            # Classification Report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred_test))
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"🏆 Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        print("="*60)
        
        # Save comparison plot
        self.plot_model_comparison(results)
        
        # Plot confusion matrix for best model
        self.plot_confusion_matrix(
            self.y_test, 
            results[best_model_name]['predictions'],
            best_model_name
        )
        
        # Plot ROC curve
        if hasattr(self.best_model, 'predict_proba'):
            y_pred_proba = self.best_model.predict_proba(X_test_tfidf)[:, 1]
            self.plot_roc_curve(self.y_test, y_pred_proba, best_model_name)
        
        return results, best_model_name
    
    def plot_model_comparison(self, results):
        """Model comparison visualization"""
        metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[model][metric] for model in model_names]
            
            bars = ax.bar(model_names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Model comparison plot saved!")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'models/confusion_matrix_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved for {model_name}!")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """ROC curve visualization"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#4ECDC4', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}', fontweight='bold', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'models/roc_curve_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved for {model_name}!")
        plt.close()
    
    def save_model(self):
        """Model aur vectorizer ko save karna"""
        print("\n" + "="*60)
        print("STEP 6: Saving Model & Vectorizer")
        print("="*60)
        
        # Save model
        joblib.dump(self.best_model, 'models/fake_news_model.pkl')
        print("✓ Model saved: models/fake_news_model.pkl")
        
        # Save vectorizer
        joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        print("✓ Vectorizer saved: models/tfidf_vectorizer.pkl")
        
        print("\n✅ All files saved successfully!")
    
    def run_complete_pipeline(self):
        """Complete pipeline ek saath run karna"""
        print("\n" + "🚀"*30)
        print("FAKE NEWS DETECTION - COMPLETE TRAINING PIPELINE")
        print("🚀"*30)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Create TF-IDF features
        X_train_tfidf, X_test_tfidf = self.create_tfidf_features()
        
        # Step 5: Train models
        results, best_model_name = self.train_models(X_train_tfidf, X_test_tfidf)
        
        # Step 6: Save model
        self.save_model()
        
        print("\n" + "🎉"*30)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉"*30)
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"✓ Model saved in 'models/' folder")
        print(f"✓ All visualizations saved!")
        print("\nNext step: Run 'python app.py' to start Flask server")


if __name__ == "__main__":
    # Create models directory if not exists
    import os
    os.makedirs('models', exist_ok=True)
    
    # Initialize and run
    detector = FakeNewsDetector(dataset_path='data/news.csv')
    detector.run_complete_pipeline()