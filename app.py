import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

# Create flask app
flask_app = Flask(__name__)
models = {
    'knn': pickle.load(open("knn1.pkl", "rb")),
    'fdt': pickle.load(open("fdt1.pkl", "rb")),
    'mlp': pickle.load(open("mlp1.pkl", "rb"))
}

@flask_app.route("/")
def home():
    return render_template("index.html", prediction_text="", temperature_input="", humidity_input="",
                           pressure_input="", wind_speed_input="", wind_bearing_input="",
                           visibility_input="", month_input="", hour_input="")

@flask_app.route("/predict", methods=["POST"])
def predict():
    selected_model = request.form.get('model_selection', 'knn')  # Default to 'knn' if model_selection is missing
    model = models[selected_model]

    float_features = [float(x) for x in request.form.values() if x != selected_model]
    # Add a default value of 0 for the missing feature
    float_features.append(0.0)
    features = [np.array(float_features)]
    prediction = model.predict(features)
    temperature_input = request.form.get('Temperature', '')
    humidity_input = request.form.get('Humidity', '')
    pressure_input = request.form.get('Pressure', '')
    wind_speed_input = request.form.get('Wind_Speed', '')
    wind_bearing_input = request.form.get('Wind_Bearing', '')
    visibility_input = request.form.get('Visibility', '')
    month_input = request.form.get('Month', '')
    hour_input = request.form.get('Hour', '')

    response = {
        "prediction_text": prediction[0],
        "temperature_input": temperature_input,
        "humidity_input": humidity_input,
        "pressure_input": pressure_input,
        "wind_speed_input": wind_speed_input,
        "wind_bearing_input": wind_bearing_input,
        "visibility_input": visibility_input,
        "month_input": month_input,
        "hour_input": hour_input
    }
    return jsonify(response)


if __name__ == "__main__":
    flask_app.run(debug=True)
