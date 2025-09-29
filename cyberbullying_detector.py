"""
Cyber Bullying Detection System
A comprehensive machine learning system for detecting cyber bullying in text data.
"""

import pandas as pd
import numpy as np
import re
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Text processing imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Machine learning imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK data download may have failed. Please ensure internet connection.")

class CyberBullyingDetector:
    """
    A comprehensive cyber bullying detection system with multiple ML models.
    """
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.best_model = None
        self.best_vectorizer = None
        self.label_encoder = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def preprocess_text(self, text):
        """
        Advanced text preprocessing for cyber bullying detection.
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming/lemmatization
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                # Use lemmatization for better results
                processed_tokens.append(self.lemmatizer.lemmatize(token))
        
        return ' '.join(processed_tokens)
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment-based features for better classification.
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return [scores['pos'], scores['neg'], scores['neu'], scores['compound']]
    
    def load_data(self, file_path):
        """
        Load and preprocess the dataset.
        """
        print("Loading dataset...")
        self.df = pd.read_csv(file_path)
        print(f"Dataset loaded: {self.df.shape}")
        
        # Check for missing values
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Preprocess text
        print("Preprocessing text...")
        self.df['processed_text'] = self.df['tweet_text'].apply(self.preprocess_text)
        
        # Extract sentiment features
        print("Extracting sentiment features...")
        sentiment_features = self.df['processed_text'].apply(self.extract_sentiment_features)
        self.df[['pos_sentiment', 'neg_sentiment', 'neu_sentiment', 'compound_sentiment']] = \
            pd.DataFrame(sentiment_features.tolist(), index=self.df.index)
        
        # Prepare features and target
        self.X = self.df['processed_text']
        self.y = self.df['cyberbullying_type']
        
        print("Data preprocessing complete!")
        return self.df
    
    def create_vectorizers(self):
        """
        Create different vectorizers for text feature extraction.
        """
        self.vectorizers = {
            'count': CountVectorizer(max_features=10000, ngram_range=(1, 2)),
            'tfidf': TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
            'count_bigram': CountVectorizer(max_features=10000, ngram_range=(2, 2)),
            'tfidf_bigram': TfidfVectorizer(max_features=10000, ngram_range=(2, 2))
        }
    
    def create_models(self):
        """
        Create different machine learning models.
        """
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
    
    def train_models(self, test_size=0.2, random_state=42):
        """
        Train all models with different vectorizers and evaluate performance.
        """
        print("Training models...")
        results = {}
        
        for vec_name, vectorizer in self.vectorizers.items():
            print(f"\n--- Training with {vec_name} vectorizer ---")
            
            # Transform text data
            X_vectorized = vectorizer.fit_transform(self.X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            
            vec_results = {}
            
            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                vec_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'model': model
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            results[vec_name] = vec_results
        
        # Find best model
        best_score = 0
        for vec_name, vec_results in results.items():
            for model_name, metrics in vec_results.items():
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    self.best_vectorizer = self.vectorizers[vec_name]
                    self.best_model = metrics['model']
                    self.best_vectorizer_name = vec_name
                    self.best_model_name = model_name
        
        print(f"\nBest model: {self.best_model_name} with {self.best_vectorizer_name}")
        print(f"Best F1 score: {best_score:.4f}")
        
        return results
    
    def predict(self, text):
        """
        Predict cyber bullying type for given text.
        """
        if self.best_model is None or self.best_vectorizer is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Transform text
        X_vectorized = self.best_vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.best_model.predict(X_vectorized)[0]
        probability = self.best_model.predict_proba(X_vectorized)[0]
        
        return prediction, probability
    
    def save_model(self, filepath='cyberbullying_model.pkl'):
        """
        Save the trained model and vectorizer.
        """
        model_data = {
            'model': self.best_model,
            'vectorizer': self.best_vectorizer,
            'vectorizer_name': self.best_vectorizer_name,
            'model_name': self.best_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='cyberbullying_model.pkl'):
        """
        Load a pre-trained model and vectorizer.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_vectorizer = model_data['vectorizer']
        self.best_vectorizer_name = model_data['vectorizer_name']
        self.best_model_name = model_data['model_name']
        
        print(f"Model loaded from {filepath}")
    
    def get_detailed_report(self, y_true, y_pred):
        """
        Generate detailed classification report.
        """
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': cm
        }


def main():
    """
    Main function to demonstrate the cyber bullying detection system.
    """
    print("Cyber Bullying Detection System")
    print("="*40)
    
    # Initialize detector
    detector = CyberBullyingDetector()
    
    # Load data
    detector.load_data('cyberbullying_tweets(ML).csv')
    
    # Create vectorizers and models
    detector.create_vectorizers()
    detector.create_models()
    
    # Train models
    results = detector.train_models()
    
    # Save the best model
    detector.save_model()
    
    # Test with sample texts
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE TEXTS")
    print("="*50)
    
    test_texts = [
        "You are so stupid and worthless!",
        "Have a great day! Looking forward to our meeting.",
        "I hate you! You should just disappear!",
        "Thanks for the help, really appreciate it!",
        "You're such a loser, nobody likes you!"
    ]
    
    for text in test_texts:
        prediction, probability = detector.predict(text)
        print(f"\nText: '{text}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probability):.4f}")


if __name__ == "__main__":
    main()
