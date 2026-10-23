"""
Dataset Checker and Fixer
Ye script tumhare CSV ko check karega aur fix karega
"""

import pandas as pd
import os

def check_and_fix_dataset():
    """
    Dataset ko check aur fix karta hai
    """
    print("="*60)
    print("DATASET CHECKER & FIXER")
    print("="*60)
    
    dataset_path = 'data/news.csv'
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ File not found: {dataset_path}")
        print("\n📌 Solutions:")
        print("1. Check if 'data' folder exists")
        print("2. Check if dataset file is named 'news.csv'")
        print("3. Download dataset from Kaggle")
        return
    
    print(f"\n✓ File found: {dataset_path}")
    print(f"File size: {os.path.getsize(dataset_path) / 1024 / 1024:.2f} MB")
    
    # Try different loading methods
    print("\n" + "-"*60)
    print("Attempting to load dataset...")
    print("-"*60)
    
    df = None
    
    # Method 1: Standard
    try:
        print("\nMethod 1: Standard UTF-8...")
        df = pd.read_csv(dataset_path, encoding='utf-8')
        print("✓ Success with UTF-8!")
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}")
    
    # Method 2: Latin-1
    if df is None:
        try:
            print("\nMethod 2: Latin-1 encoding...")
            df = pd.read_csv(dataset_path, encoding='latin-1')
            print("✓ Success with Latin-1!")
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
    
    # Method 3: Skip bad lines
    if df is None:
        try:
            print("\nMethod 3: Skipping bad lines...")
            df = pd.read_csv(dataset_path, 
                           encoding='utf-8',
                           on_bad_lines='skip',
                           engine='python')
            print("✓ Success with skip method!")
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
    
    # Method 4: ISO-8859-1
    if df is None:
        try:
            print("\nMethod 4: ISO-8859-1 encoding...")
            df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
            print("✓ Success with ISO-8859-1!")
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
    
    # Method 5: With error handling
    if df is None:
        try:
            print("\nMethod 5: Error handling mode...")
            df = pd.read_csv(dataset_path, 
                           encoding='utf-8',
                           on_bad_lines='skip',
                           engine='python',
                           sep=',',
                           quotechar='"',
                           low_memory=False)
            print("✓ Success with error handling!")
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
    
    if df is None:
        print("\n❌ All methods failed!")
        print("\n📌 Manual Solutions:")
        print("1. Open CSV in Excel/Notepad")
        print("2. Check for corrupted lines")
        print("3. Re-save as clean CSV")
        print("4. Try different dataset from Kaggle")
        return
    
    # Dataset loaded successfully
    print("\n" + "="*60)
    print("DATASET INFO")
    print("="*60)
    
    print(f"\n✓ Shape: {df.shape}")
    print(f"✓ Rows: {df.shape[0]:,}")
    print(f"✓ Columns: {df.shape[1]}")
    
    print(f"\n📊 Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col} ({df[col].dtype})")
    
    print(f"\n📈 Data Types:")
    print(df.dtypes)
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values!")
    
    print(f"\n🔢 Sample Data:")
    print(df.head(3))
    
    # Check for required columns
    print("\n" + "="*60)
    print("COLUMN VALIDATION")
    print("="*60)
    
    text_columns = ['text', 'content', 'article', 'news', 'title']
    label_columns = ['label', 'class', 'target', 'is_fake', 'fake']
    
    found_text = None
    found_label = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(tc in col_lower for tc in text_columns):
            found_text = col
        if any(lc in col_lower for lc in label_columns):
            found_label = col
    
    if found_text:
        print(f"✓ Text column found: '{found_text}'")
        print(f"  Sample: {df[found_text].iloc[0][:100]}...")
    else:
        print("❌ No text column found!")
        print(f"   Available: {list(df.columns)}")
        print(f"   Expected: {text_columns}")
    
    if found_label:
        print(f"\n✓ Label column found: '{found_label}'")
        print(f"  Unique values: {df[found_label].unique()}")
        print(f"  Distribution:\n{df[found_label].value_counts()}")
    else:
        print("\n❌ No label column found!")
        print(f"   Available: {list(df.columns)}")
        print(f"   Expected: {label_columns}")
    
    # Try to fix and save
    if found_text and found_label:
        print("\n" + "="*60)
        print("CREATING CLEAN DATASET")
        print("="*60)
        
        # Create clean dataset
        clean_df = pd.DataFrame()
        clean_df['text'] = df[found_text]
        clean_df['label'] = df[found_label]
        
        # Remove nulls
        clean_df = clean_df.dropna()
        
        # Convert labels to binary if needed
        if clean_df['label'].dtype == 'object':
            unique_labels = clean_df['label'].unique()
            if len(unique_labels) == 2:
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                clean_df['label'] = clean_df['label'].map(label_map)
                print(f"\n✓ Labels converted: {label_map}")
        
        # Save clean dataset
        output_path = 'data/news_clean.csv'
        clean_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n✅ Clean dataset saved: {output_path}")
        print(f"✓ Shape: {clean_df.shape}")
        print(f"✓ Ready for training!")
        
        print("\n📌 Next Step:")
        print("Update model_training.py line 40:")
        print("  dataset_path='data/news_clean.csv'")
    else:
        print("\n❌ Cannot create clean dataset")
        print("Please check your CSV structure manually")
    
    print("\n" + "="*60)
    print("CHECKER COMPLETED")
    print("="*60)


if __name__ == "__main__":
    check_and_fix_dataset()