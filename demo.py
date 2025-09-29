"""
Simple demonstration of the Cyber Bullying Detection System
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK data download may have failed. Please ensure internet connection.")

def preprocess_text(text):
    """Simple text preprocessing"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    tokens = nltk.word_tokenize(text)
    processed_tokens = []
    
    for token in tokens:
        if len(token) > 2 and token not in stop_words:
            processed_tokens.append(stemmer.stem(token))
    
    return ' '.join(processed_tokens)

def main():
    print("Cyber Bullying Detection System - Demo")
    print("=" * 40)
    
    # Load data
    print("Loading dataset...")
    try:
        df = pd.read_csv('cyberbullying_tweets(ML).csv')
        print(f"Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset file not found!")
        return
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['tweet_text'].apply(preprocess_text)
    
    # Prepare features and target
    X = df['processed_text']
    y = df['cyberbullying_type']
    
    # Create vectorizer
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
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
        processed = preprocess_text(text)
        
        # Transform
        X_test_text = vectorizer.transform([processed])
        
        # Predict
        prediction = model.predict(X_test_text)[0]
        probability = model.predict_proba(X_test_text)[0]
        confidence = max(probability)
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
    
    print(f"\n" + "=" * 40)
    print("DEMO COMPLETED")
    print("=" * 40)

if __name__ == "__main__":
    main()
