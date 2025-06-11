from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load(r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\model\disease_model.pkl")
vectorizer = joblib.load(r"C:\Users\jatin\OneDrive\Desktop\python\AI disease predictor\model\symptom_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        user_input = request.form["symptoms"]
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
