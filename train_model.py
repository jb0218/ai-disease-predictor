import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load data
df = pd.read_csv(r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\data\disease_dataset.csv")

# Vectorize symptoms
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["symptoms"])
y = df["disease"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model & vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\model/disease_model.pkl")
joblib.dump(vectorizer, r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\model/symptom_vectorizer.pkl")

print("✅ Model and vectorizer saved.")
