from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load model & scalers
# ----------------------------
model = load_model("best_energy_model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

TIME_STEPS = 14

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/smart_home_energy_complete_dataset.csv")

# ----------------------------
# Fix timestamp
# ----------------------------
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H.%M",
    errors="coerce"
)
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

# ----------------------------
# Feature columns (EXACT training features)
# ----------------------------
FEATURE_COLUMNS = [
    col for col in df.columns
    if col != "total_energy_consumption"
]

NUM_FEATURES = len(FEATURE_COLUMNS)
print("DEBUG NUM_FEATURES:", NUM_FEATURES)

# ----------------------------
# Device columns (UI only)
# ----------------------------
DEVICE_COLUMNS = [
    "fridge",
    "wine_cellar",
    "garage_door",
    "microwave",
    "living_room_appliances"
]

# ----------------------------
# Smart tips
# ----------------------------
def smart_tip(device):
    tips = {
        "fridge": "Avoid frequent door opening to reduce cooling loss.",
        "wine_cellar": "Maintain stable temperature to save energy.",
        "garage_door": "Reduce unnecessary open-close cycles.",
        "microwave": "Prefer microwave for small meals instead of oven.",
        "living_room_appliances": "Turn off devices instead of standby."
    }
    return tips.get(device, "Monitor usage for energy efficiency.")

# ----------------------------
# Device energy share
# ----------------------------
def device_share(device):
    recent = df[DEVICE_COLUMNS].tail(100)
    total = recent.sum(axis=1).mean()
    return 0.0 if total == 0 else recent[device].mean() / total

# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/")
def home():
    recent_data = df[FEATURE_COLUMNS].tail(TIME_STEPS).values.tolist()

    print("DEBUG recent_data shape:", np.array(recent_data).shape)

    return render_template(
        "index.html",
        devices=DEVICE_COLUMNS,
        recent_data=recent_data
    )

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        device = data.get("device")
        horizon = data.get("horizon")
        features = np.array(data.get("features"), dtype=float)

        if features.shape != (TIME_STEPS, NUM_FEATURES):
            return jsonify({
                "error": f"Expected ({TIME_STEPS}, {NUM_FEATURES}), got {features.shape}"
            }), 400

        steps_map = {"hour": 1, "week": 7, "month": 30}
        steps = steps_map.get(horizon, 1)

        current_seq = features.copy()
        preds = []

        for _ in range(steps):
            scaled = scaler_X.transform(current_seq)
            scaled = scaled.reshape(1, TIME_STEPS, NUM_FEATURES)

            pred_scaled = model.predict(scaled, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            pred = max(float(pred), 0.0)

            preds.append(pred)

            # Roll sequence (NO fake values)
            current_seq = np.vstack([current_seq[1:], current_seq[-1]])

        # ✔ SINGLE OUTPUT VALUE
        total_energy = preds[0] if horizon == "hour" else sum(preds)

        share = device_share(device)
        device_energy = total_energy * share

        return jsonify({
            "total_energy_kwh": round(total_energy, 3),
            "device_energy_kwh": round(device_energy, 3),
            "tip": smart_tip(device)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
