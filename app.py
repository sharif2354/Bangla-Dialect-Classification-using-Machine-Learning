from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and vectorizer
model = pickle.load(open("language_model.pkl", "rb"))  # Replace with your saved model
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Replace with your saved vectorizer

@app.route("/")
def home():
    return render_template("index.html")  # Frontend HTML file

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form["text"]  # Input text from the user
    processed_data = vectorizer.transform([data])
    prediction = model.predict(processed_data)
    return jsonify({"dialect": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
