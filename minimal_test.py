import pandas as pd
import sys

print("Starting minimal test...")
sys.stdout.flush()

try:
    df = pd.read_csv('cyberbullying_tweets(ML).csv')
    print(f"Dataset loaded successfully: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    sys.stdout.flush()
except Exception as e:
    print(f"Error: {e}")
    sys.stdout.flush()

print("Test completed!")
sys.stdout.flush()
