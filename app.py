import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
flask_app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@flask_app.route("/")
def Home():
    return render_template("index.html")

# Prediction route
@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get input features from form and convert to float
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    # Make prediction
    prediction = model.predict(features)
    predicted_crop = prediction[0]  # Get string from array

    # Render the result
    return render_template("index.html", prediction_text=f"The Predicted Crop is: {predicted_crop}")

# Run the app
if __name__ == "__main__":
    flask_app.run(debug=True)
