"""
Quick System Test - Check if everything is working
"""

import os
import sys

print("="*60)
print("FAKE NEWS DETECTION SYSTEM - DIAGNOSTIC TEST")
print("="*60)

# Test 1: Check Python version
print("\n[1/7] Checking Python version...")
print(f"✓ Python {sys.version}")

# Test 2: Check required files
print("\n[2/7] Checking project files...")
required_files = ['app.py', 'model_training.py', 'templates/index.html', 
                  'static/css/style.css', 'static/js/script.js']

for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING!")

# Test 3: Check packages
print("\n[3/7] Checking installed packages...")
required_packages = ['flask', 'pandas', 'numpy', 'sklearn', 'nltk', 'joblib']

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - NOT INSTALLED!")
        print(f"   Install: pip install {package}")

# Test 4: Check data folder
print("\n[4/7] Checking data folder...")
if os.path.exists('data'):
    print("✓ data/ folder exists")
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if csv_files:
        print(f"✓ Found {len(csv_files)} CSV file(s): {csv_files}")
    else:
        print("✗ No CSV files found in data/")
else:
    print("✗ data/ folder missing!")

# Test 5: Check models folder
print("\n[5/7] Checking models folder...")
if os.path.exists('models'):
    print("✓ models/ folder exists")
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    if model_files:
        print(f"✓ Found {len(model_files)} model file(s): {model_files}")
    else:
        print("⚠  No .pkl files found - Need to train model!")
        print("   Run: python model_training.py")
else:
    print("✗ models/ folder missing!")
    os.makedirs('models', exist_ok=True)
    print("✓ Created models/ folder")

# Test 6: Try loading model
print("\n[6/7] Testing model loading...")
try:
    import joblib
    model = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("✓ Model loaded successfully!")
    print("✓ Vectorizer loaded successfully!")
except FileNotFoundError:
    print("✗ Model files not found!")
    print("   Solution: Run 'python model_training.py'")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")

# Test 7: Test Flask
print("\n[7/7] Testing Flask setup...")
try:
    from flask import Flask
    test_app = Flask(__name__)
    print("✓ Flask import successful!")
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if os.path.exists('models/fake_news_model.pkl'):
        print("\n✅ System is READY!")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open: http://127.0.0.1:5000")
    else:
        print("\n⚠️  Model not trained!")
        print("\nNext steps:")
        print("1. Run: python model_training.py")
        print("2. Wait for training to complete (~5-10 minutes)")
        print("3. Then run: python app.py")
except ImportError:
    print("✗ Flask not installed!")
    print("   Install: pip install Flask==2.3.3")

print("\n" + "="*60)