from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
SPECIES = ["Setosa", "Versicolor", "Virginica"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"]),
        ]
    except (KeyError, ValueError):
        return render_template("index.html", error="Please enter valid numeric values for all fields.")

    prediction = model.predict(np.array([features]))[0]
    species = SPECIES[prediction]

    return render_template("result.html", species=species, features=features)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
