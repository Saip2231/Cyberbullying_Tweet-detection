"""
Flask Web Application for Cyber Bullying Detection
A user-friendly web interface for real-time cyber bullying detection.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import os
from cyberbullying_detector import CyberBullyingDetector
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global variables
detector = None
model_loaded = False

def load_or_train_model():
    """
    Load existing model or train a new one if not available.
    """
    global detector, model_loaded
    
    detector = CyberBullyingDetector()
    
    # Try to load existing model
    if os.path.exists('cyberbullying_model.pkl'):
        try:
            detector.load_model('cyberbullying_model.pkl')
            model_loaded = True
            print("Model loaded successfully!")
        except:
            print("Failed to load existing model. Training new model...")
            model_loaded = False
    else:
        print("No existing model found. Training new model...")
        model_loaded = False
    
    if not model_loaded:
        try:
            # Load and preprocess data
            detector.load_data('cyberbullying_tweets(ML).csv')
            
            # Create vectorizers and models
            detector.create_vectorizers()
            detector.create_models()
            
            # Train models
            detector.train_models()
            
            # Save the trained model
            detector.save_model('cyberbullying_model.pkl')
            model_loaded = True
            print("Model trained and saved successfully!")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            model_loaded = False

@app.route('/')
def index():
    """
    Main page with text input for cyber bullying detection.
    """
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for cyber bullying prediction.
    """
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please try again later.',
            'prediction': None,
            'confidence': None
        })
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({
                'error': 'Please enter some text to analyze.',
                'prediction': None,
                'confidence': None
            })
        
        # Get prediction
        prediction, probability = detector.predict(text)
        confidence = max(probability)
        
        # Get class probabilities
        classes = detector.best_model.classes_
        probabilities = dict(zip(classes, probability))
        
        return jsonify({
            'error': None,
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error during prediction: {str(e)}',
            'prediction': None,
            'confidence': None
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch prediction of multiple texts.
    """
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please try again later.',
            'results': None
        })
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'error': 'Please provide texts to analyze.',
                'results': None
            })
        
        results = []
        for text in texts:
            if text.strip():
                prediction, probability = detector.predict(text)
                confidence = max(probability)
                results.append({
                    'text': text,
                    'prediction': prediction,
                    'confidence': float(confidence)
                })
        
        return jsonify({
            'error': None,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error during batch prediction: {str(e)}',
            'results': None
        })

@app.route('/stats')
def stats():
    """
    Display model statistics and performance metrics.
    """
    if not model_loaded:
        return render_template('stats.html', error='Model not loaded')
    
    try:
        # Get model information
        model_info = {
            'model_name': detector.best_model_name,
            'vectorizer_name': detector.best_vectorizer_name,
            'classes': detector.best_model.classes_.tolist() if hasattr(detector.best_model, 'classes_') else []
        }
        
        return render_template('stats.html', model_info=model_info)
        
    except Exception as e:
        return render_template('stats.html', error=f'Error loading stats: {str(e)}')

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain the model with current data.
    """
    global model_loaded
    
    try:
        # Load and preprocess data
        detector.load_data('cyberbullying_tweets(ML).csv')
        
        # Create vectorizers and models
        detector.create_vectorizers()
        detector.create_models()
        
        # Train models
        results = detector.train_models()
        
        # Save the trained model
        detector.save_model('cyberbullying_model.pkl')
        model_loaded = True
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retraining model: {str(e)}'
        })

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
