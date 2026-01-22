let energyChart = null;
let deviceChart = null;

(function () {
    'use strict';

    const deviceSelect = document.getElementById('device-select');
    const horizonSelect = document.getElementById('horizon-select');
    const predictBtn = document.getElementById('predict-btn');
    const loadingIndicator = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');
    const totalEnergyEl = document.getElementById('total-energy');
    const deviceEnergyEl = document.getElementById('device-energy');
    const energyTipEl = document.getElementById('energy-tip');
    const rawResponseEl = document.getElementById('raw-response');

    const API_ENDPOINT = '/predict';

    function showElement(el) {
        if (el) el.classList.remove('hidden');
    }

    function hideElement(el) {
        if (el) el.classList.add('hidden');
    }

    function displayError(msg) {
        hideElement(loadingIndicator);
        hideElement(resultsContainer);
        errorMessage.textContent = msg;
        showElement(errorMessage);
    }

    function clearError() {
        hideElement(errorMessage);
        errorMessage.textContent = '';
    }

    function formatEnergy(v) {
        return (typeof v === 'number' && !isNaN(v))
            ? v.toFixed(2) + ' kWh'
            : '-- kWh';
    }

    function validateInputs() {
        if (!deviceSelect.value) {
            displayError('Please select a device.');
            return null;
        }
        if (!horizonSelect.value) {
            displayError('Please select a prediction horizon.');
            return null;
        }
        if (!window.recent_data) {
            displayError('Recent data not available. Refresh page.');
            return null;
        }
        return {
            device: deviceSelect.value,
            horizon: horizonSelect.value
        };
    }

    async function makePrediction(device, horizon) {
        try {
            const res = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    device,
                    horizon,
                    features: window.recent_data
                })
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error || 'Server error');
            }

            const data = await res.json();
            displayResults(data);

        } catch (e) {
            console.error(e);
            displayError(e.message);
        }
    }

    function displayResults(data) {
        hideElement(loadingIndicator);
        clearError();

        totalEnergyEl.textContent = formatEnergy(data.total_energy_kwh);
        deviceEnergyEl.textContent = formatEnergy(data.device_energy_kwh);
        energyTipEl.textContent = data.tip || '--';
        rawResponseEl.textContent = JSON.stringify(data, null, 2);
        showElement(resultsContainer);

        /* -------- DEVICE TIME-SERIES CHART -------- */
        const ctx1 = document.getElementById("energyChart").getContext("2d");

        if (energyChart) energyChart.destroy();

        const ratio = data.total_energy_kwh > 0
            ? data.device_energy_kwh / data.total_energy_kwh
            : 0;

        const deviceSeries = data.values.map(v => +(v * ratio).toFixed(3));

        energyChart = new Chart(ctx1, {
            type: "bar",
            data: {
                labels: data.labels,
                datasets: [{
                    label: "Device Energy (kWh)",
                    data: deviceSeries,
                    backgroundColor: "#16a34a"
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });

        /* -------- MULTI DEVICE COMPARISON CHART -------- */
        if (!data.device_comparison) return;

        const ctx2 = document
            .getElementById("deviceComparisonChart")
            .getContext("2d");

        if (deviceChart) deviceChart.destroy();

        const labels = Object.keys(data.device_comparison)
            .map(d => d.replaceAll("_", " ").toUpperCase());

        const values = Object.values(data.device_comparison);

        deviceChart = new Chart(ctx2, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: "Energy (kWh)",
                    data: values,
                    backgroundColor: [
                        "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"
                    ]
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { display: false } }
            }
        });
    }

    function handlePredictClick() {
        clearError();
        hideElement(resultsContainer);

        const inputs = validateInputs();
        if (!inputs) return;

        showElement(loadingIndicator);
        predictBtn.disabled = true;

        makePrediction(inputs.device, inputs.horizon)
            .finally(() => {
                predictBtn.disabled = false;
                hideElement(loadingIndicator);
            });
    }

    function init() {
        predictBtn?.addEventListener('click', handlePredictClick);
        console.log("Dashboard initialized");
    }

    document.readyState === 'loading'
        ? document.addEventListener('DOMContentLoaded', init)
        : init();
})();
