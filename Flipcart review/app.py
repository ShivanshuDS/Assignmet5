from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review = clean_text(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    confidence = max(model.predict_proba(review_vector)[0])

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence*100,2)
    )


if __name__ == "__main__":
    app.run(debug=True)