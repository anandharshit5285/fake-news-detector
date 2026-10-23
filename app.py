"""
Flask Backend for Fake News Detection
Ye file web server run karegi aur predictions handle karegi
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
print("Loading model and vectorizer...")
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
print("✓ Model and vectorizer loaded successfully!")

# Initialize preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Text preprocessing - same as training
    """
    # Convert to string and lowercase
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
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)


def predict_news(text):
    """
    News prediction with confidence score
    """
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities) * 100)
    else:
        confidence = 85.0  # Default confidence
    
    # Result
    result = {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'confidence': round(confidence, 2),
        'label': int(prediction)
    }
    
    return result


@app.route('/')
def home():
    """Homepage"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    """
    try:
        # Get text from request
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text or len(news_text.strip()) < 10:
            return jsonify({
                'error': 'Please enter at least 10 characters of news text'
            }), 400
        
        # Make prediction
        result = predict_news(news_text)
        
        # Add additional info
        result['text_length'] = len(news_text)
        result['word_count'] = len(news_text.split())
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Detailed analysis endpoint
    """
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Basic prediction
        result = predict_news(news_text)
        
        # Additional analysis
        cleaned = preprocess_text(news_text)
        text_features = vectorizer.transform([cleaned])
        
        # Get top TF-IDF features
        feature_names = vectorizer.get_feature_names_out()
        feature_array = text_features.toarray()[0]
        top_indices = feature_array.argsort()[-10:][::-1]
        
        top_features = [
            {
                'word': feature_names[i],
                'score': round(float(feature_array[i]), 4)
            }
            for i in top_indices if feature_array[i] > 0
        ]
        
        # Detailed analysis
        analysis = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'label': result['label'],
            'statistics': {
                'original_length': len(news_text),
                'cleaned_length': len(cleaned),
                'word_count': len(news_text.split()),
                'cleaned_word_count': len(cleaned.split()),
                'unique_words': len(set(cleaned.split()))
            },
            'top_features': top_features,
            'cleaned_text': cleaned[:500] + '...' if len(cleaned) > 500 else cleaned
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Fake News Detection Server...")
    print("="*60)
    print("\n✓ Server running at: http://127.0.0.1:5000")
    print("✓ Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)