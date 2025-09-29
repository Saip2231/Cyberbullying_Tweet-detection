"""
Test script for the Cyber Bullying Detection System
Demonstrates the functionality of the detection system.
"""

from cyberbullying_detector import CyberBullyingDetector
import time

def test_system():
    """
    Test the cyber bullying detection system with sample texts.
    """
    print("Cyber Bullying Detection System - Test Script")
    print("=" * 50)
    
    # Initialize detector
    print("Initializing detector...")
    detector = CyberBullyingDetector()
    
    # Load and train model
    print("Loading data and training model...")
    start_time = time.time()
    
    try:
        detector.load_data('cyberbullying_tweets(ML).csv')
        detector.create_vectorizers()
        detector.create_models()
        detector.train_models()
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Save model
        detector.save_model('test_model.pkl')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    # Test with sample texts
    print("\n" + "=" * 50)
    print("TESTING WITH SAMPLE TEXTS")
    print("=" * 50)
    
    test_cases = [
        {
            "text": "You are so stupid and worthless!",
            "expected": "cyberbullying",
            "description": "Direct insult"
        },
        {
            "text": "Have a great day! Looking forward to our meeting.",
            "expected": "not_cyberbullying", 
            "description": "Positive message"
        },
        {
            "text": "I hate you! You should just disappear!",
            "expected": "cyberbullying",
            "description": "Hateful message"
        },
        {
            "text": "Thanks for the help, really appreciate it!",
            "expected": "not_cyberbullying",
            "description": "Grateful message"
        },
        {
            "text": "You're such a loser, nobody likes you!",
            "expected": "cyberbullying",
            "description": "Bullying message"
        },
        {
            "text": "Great job on the presentation today!",
            "expected": "not_cyberbullying",
            "description": "Complimentary message"
        }
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        
        try:
            prediction, probability = detector.predict(test_case['text'])
            confidence = max(probability)
            
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
            
            # Check if prediction matches expected
            if prediction == test_case['expected']:
                print("✓ Correct prediction!")
                correct_predictions += 1
            else:
                print(f"✗ Incorrect prediction (expected: {test_case['expected']})")
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n" + "=" * 50)
    print(f"TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Test model loading
    print(f"\n" + "=" * 50)
    print("TESTING MODEL LOADING")
    print("=" * 50)
    
    try:
        # Create new detector instance
        new_detector = CyberBullyingDetector()
        new_detector.load_model('test_model.pkl')
        
        # Test prediction with loaded model
        test_text = "This is a test message for the loaded model."
        prediction, probability = new_detector.predict(test_text)
        confidence = max(probability)
        
        print(f"Test text: '{test_text}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print("✓ Model loading and prediction successful!")
        
    except Exception as e:
        print(f"Error during model loading test: {str(e)}")
    
    print(f"\n" + "=" * 50)
    print("TEST COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    test_system()
