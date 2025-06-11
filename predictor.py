import joblib

# Load model & vectorizer
model = joblib.load(r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\model\disease_model.pkl")
vectorizer = joblib.load(r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\model\symptom_vectorizer.pkl")

# Input symptoms
symptoms_input = input("Enter your symptoms (comma-separated):\n")

# Process & predict
X = vectorizer.transform([symptoms_input])
prediction = model.predict(X)
probs = model.predict_proba(X)

print("\n🤖 Most Likely Disease:", prediction[0])
print("📊 Prediction Probabilities:")
for i, disease in enumerate(model.classes_):
    print(f" - {disease}: {probs[0][i]*100:.2f}%")
