from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load("suv_purchase_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")
except Exception as e:
    print("Model or scaler not found:", e)
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("index.html", prediction_text="Model or Scaler not loaded. Please train your model.")

    try:
        age = int(request.form['age'])
        salary = int(request.form['salary'])

        features = np.array([[age, salary]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        result = "Customer is LIKELY to Purchase an SUV" if prediction[0] == 1 else "Customer is UNLIKELY to Purchase an SUV"

        return render_template("index.html", prediction_text=result, age_val=age, salary_val=salary)

    except Exception as e:
        return render_template("index.html", prediction_text=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True)
