// ===============================
// Wait until HTML is fully loaded
// ===============================
document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("predictBtn");

    if (btn) {
        btn.addEventListener("click", predict);
        console.log("Predict button connected");
    } else {
        console.error("Predict button NOT found (check id='predictBtn')");
    }
});

// ===============================
// Predict function
// ===============================
function predict() {
    const output = document.getElementById("output");

    if (!output) {
        console.error("Output element not found (id='output')");
        return;
    }

    output.innerText = "Predicting...";

    // -------------------------------
    // Check recent data from backend
    // -------------------------------
    if (!window.recent_data) {
        output.innerText = "Recent data not loaded from backend.";
        console.error("window.recent_data is undefined");
        return;
    }

    if (window.recent_data.length !== 14) {
        output.innerText = "Recent data length is not 14.";
        console.error("recent_data length:", window.recent_data.length);
        return;
    }

    // -------------------------------
    // Read user selections
    // -------------------------------
    const deviceEl = document.getElementById("device");
    const horizonEl = document.getElementById("horizon");

    if (!deviceEl || !horizonEl) {
        output.innerText = "Device or Horizon selector missing.";
        console.error("device or horizon element missing");
        return;
    }

    const device = deviceEl.value;
    const horizon = horizonEl.value;

    // -------------------------------
    // Ensure features are numeric
    // -------------------------------
    const features = window.recent_data.map(row =>
        row.map(v => Number(v))
    );

    // -------------------------------
    // Send request to Flask backend
    // -------------------------------
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            device: device,
            horizon: horizon,
            features: features
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            output.innerText = "Error: " + data.error;
            console.error("Backend error:", data.error);
            return;
        }

        // -------------------------------
        // Display results (MATCHES app.py)
        // -------------------------------
        if (!data.total_predictions || !data.device_predictions) {
        output.innerText = "Invalid response from backend.";
        console.error("Invalid response:", data);
        return;
    }

    output.innerText =
        "Total Energy Prediction (kWh):\n" +
        data.total_predictions.join(", ") + "\n\n" +
        device + " Energy Prediction (kWh):\n" +
        data.device_predictions.join(", ") + "\n\n" +
        "Smart Tip:\n" +
        data.tip;
        })
    .catch(err => {
        console.error("Fetch failed:", err);
        output.innerText = "Request failed. Check console.";
    });
}
