
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scalers
from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

# ----------------------------
# Load scalers
# ----------------------------
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

TIME_STEPS = 14
NUM_FEATURES = scaler_X.n_features_in_

# ----------------------------
# Rebuild model architecture
# ----------------------------
def build_lstm_model():
    model = Sequential()
    model.add(
        LSTM(
            64,
            activation="tanh",
            input_shape=(TIME_STEPS, NUM_FEATURES)
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

model = build_lstm_model()

# ----------------------------
# Load trained weights
# ----------------------------
model.load_weights("best_energy_weights.weights.h5")

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return "Energy Consumption Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        arr = np.array(data["features"])

        if arr.shape != (TIME_STEPS, NUM_FEATURES):
            return jsonify({
                "error": f"Expected shape ({TIME_STEPS}, {NUM_FEATURES}), got {arr.shape}"
            }), 400

        arr_scaled = scaler_X.transform(arr)
        arr_scaled = arr_scaled.reshape(1, TIME_STEPS, NUM_FEATURES)

        pred_scaled = model.predict(arr_scaled)
        prediction = scaler_y.inverse_transform(pred_scaled)

        pred_value = float(prediction[0][0])
        pred_value = max(pred_value, 0.0)  # energy cannot be negative

        return jsonify({
            "predicted_energy_consumption": pred_value
            })


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

TIME_STEPS = 14

app = Flask(__name__)

@app.route("/")
def home():
    return "Energy Consumption Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = np.array(data["features"])
    input_scaled = scaler_X.transform(input_data)
    input_scaled = input_scaled.reshape(
        1, TIME_STEPS, input_scaled.shape[1]
    )

    pred_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(pred_scaled)

    return jsonify({
        "predicted_energy_consumption": float(prediction[0][0])
    })

if __name__ == "__main__":
    app.run(debug=True)
