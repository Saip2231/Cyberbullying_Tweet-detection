"""
Very simple test to verify the system works
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

def simple_preprocess(text):
    """Very simple text preprocessing"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

def main():
    print("Simple Cyber Bullying Detection Test")
    print("=" * 40)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('cyberbullying_tweets(ML).csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Simple preprocessing
    print("Preprocessing text...")
    df['processed_text'] = df['tweet_text'].apply(simple_preprocess)
    
    # Take a small sample for quick testing
    sample_size = 1000
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Prepare features and target
    X = df_sample['processed_text']
    y = df_sample['cyberbullying_type']
    
    # Create vectorizer
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Naive Bayes model...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Test with sample texts
    print(f"\n" + "=" * 40)
    print("TESTING WITH SAMPLE TEXTS")
    print("=" * 40)
    
    test_texts = [
        "You are so stupid and worthless!",
        "Have a great day! Looking forward to our meeting.",
        "I hate you! You should just disappear!",
        "Thanks for the help, really appreciate it!",
        "You're such a loser, nobody likes you!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        
        # Preprocess
        processed = simple_preprocess(text)
        
        # Transform
        X_test_text = vectorizer.transform([processed])
        
        # Predict
        prediction = model.predict(X_test_text)[0]
        probability = model.predict_proba(X_test_text)[0]
        confidence = max(probability)
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
    
    print(f"\n" + "=" * 40)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 40)

if __name__ == "__main__":
    main()
