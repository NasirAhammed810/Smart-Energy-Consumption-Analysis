"""
Smart Energy Forecasting System - Flask Backend
==============================================
Author: Shaik Nasir Ahammed
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# ----------------------------------
# Flask App
# ----------------------------------
app = Flask(__name__)

# ----------------------------------
# Paths
# ----------------------------------
MODEL_PATH = "best_energy_model.h5"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
DATA_PATH = "data/smart_home_energy_complete_dataset.csv"

# ----------------------------------
# Globals
# ----------------------------------
model = None
scaler_X = None
scaler_y = None

# ----------------------------------
# Safe Artifact Loader
# ----------------------------------
def load_artifacts():
    global model, scaler_X, scaler_y

    # ---- Load Keras model safely ----
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        print("✅ ML model loaded successfully")
    except Exception as e:
        model = None
        print("⚠️ Model not loaded, fallback mode enabled")
        print("   Reason:", e)

    # ---- Load scalers using joblib ONLY ----
    try:
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("✅ Scalers loaded successfully")
    except Exception as e:
        scaler_X = None
        scaler_y = None
        print("⚠️ Scalers not loaded, fallback mode enabled")
        print("   Reason:", e)

# ----------------------------------
# Load dataset
# ----------------------------------
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    return df

df = load_dataset()

# ----------------------------------
# Utility functions
# ----------------------------------
def get_device_multiplier(device):
    return {
        "fridge": 0.18,
        "wine_cellar": 0.12,
        "garage_door": 0.05,
        "microwave": 0.15,
        "living_room_appliances": 0.25
    }.get(device, 0.15)

def get_horizon_multiplier(horizon):
    return {
        "hour": 1,
        "week": 24 * 7,
        "month": 24 * 30
    }.get(horizon, 1)

def get_energy_tip(device):
    tips = {
        "fridge": "Avoid frequent door opening to reduce cooling loss.",
        "wine_cellar": "Maintain stable temperature for efficiency.",
        "garage_door": "Reduce unnecessary open-close cycles.",
        "microwave": "Prefer microwave for small meals.",
        "living_room_appliances": "Turn off devices instead of standby."
    }
    return tips.get(device, "Monitor usage for energy efficiency.")

# ----------------------------------
# Core prediction logic
# ----------------------------------
def predict_energy(features, device, horizon):
    base_energy = 2.5  # fallback base (kWh/hour)

    # ---- ML prediction (if available) ----
    # ---- ML prediction (if available) ----
   # ---- ML prediction (FINAL FIX FOR LSTM) ----
    if model and scaler_X and scaler_y:
        try:
            # Extract numeric values only
            numeric_features = []
            for v in features.values():
                try:
                    numeric_features.append(float(v))
                except:
                    pass
    
            numeric_features = np.array(numeric_features)
    
            expected_features = scaler_X.n_features_in_      # 22
            TIME_STEPS = model.input_shape[1]                # 14
    
            # Pad / trim feature vector
            if numeric_features.shape[0] < expected_features:
                padded = np.zeros(expected_features)
                padded[:numeric_features.shape[0]] = numeric_features
                feature_row = padded
            else:
                feature_row = numeric_features[:expected_features]
    
            # 🔑 CRITICAL FIX: create (14, 22) sequence
            sequence = np.tile(feature_row, (TIME_STEPS, 1))
    
            # Scale features
            sequence_scaled = scaler_X.transform(sequence)
    
            # Final LSTM input shape → (1, 14, 22)
            X_final = sequence_scaled.reshape(1, TIME_STEPS, expected_features)
    
            y_scaled = model.predict(X_final, verbose=0)
            base_energy = float(scaler_y.inverse_transform(y_scaled)[0][0])
    
        except Exception as e:
            print("⚠️ ML prediction failed, using fallback:", e)




    device_mult = get_device_multiplier(device)
    
    # ----------------------------------
    # Horizon-wise DEVICE chart data
    # ----------------------------------
    if horizon == "hour":
        labels = ["Next Hour"]
        total_values = [base_energy]
        values = [round(base_energy * device_mult, 3)]
    
    elif horizon == "week":
        labels = [f"Day {i+1}" for i in range(7)]
        total_values = [base_energy * 24] * 7
        values = [round(v * device_mult, 3) for v in total_values]
    
    elif horizon == "month":
        labels = [f"Day {i+1}" for i in range(30)]
        total_values = [base_energy * 24] * 30
        values = [round(v * device_mult, 3) for v in total_values]
    
    else:
        labels = ["Next Hour"]
        total_values = [base_energy]
        values = [round(base_energy * device_mult, 3)]


    # ----------------------------------
    # Aggregated values (for cards)
    # ----------------------------------
    total_energy = round(sum(values), 3)
    device_energy = round(total_energy * device_mult, 3)

   # -------------------------------
    # MULTI-DEVICE COMPARISON DATA
    # -------------------------------
    all_devices = [
        "fridge",
        "wine_cellar",
        "garage_door",
        "microwave",
        "living_room_appliances"
    ]
    
    device_comparison = {}
    for d in all_devices:
        device_comparison[d] = round(
            total_energy * get_device_multiplier(d), 3
        )
    
    return {
        "total_energy_kwh": total_energy,
        "device_energy_kwh": device_energy,
        "labels": labels,
        "values": values,  # single-device chart (already working)
        "device_comparison": device_comparison,  # ✅ NEW
        "tip": get_energy_tip(device)
    }

# ----------------------------------
# Routes
# ----------------------------------
@app.route("/")
def index():
    # Provide last row of dataset to frontend
    if df is not None:
        recent_data = df.iloc[-1].drop("total_energy_consumption", errors="ignore").to_dict()
    else:
        recent_data = {}

    return render_template("index.html", recent_data=recent_data)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        device = data.get("device")
        horizon = data.get("horizon")
        features = data.get("features")

        if not device or not horizon or not features:
            return jsonify({"error": "Invalid input"}), 400

        result = predict_energy(features, device, horizon)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "scalers_loaded": scaler_X is not None and scaler_y is not None
    })

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    print("🚀 Smart Energy Forecasting System")
    load_artifacts()
    print("🌐 Server running at http://localhost:5000")
    app.run(debug=True)
