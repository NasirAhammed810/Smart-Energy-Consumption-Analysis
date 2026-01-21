# Smart-Energy-Consumption-Analysis
🏡 Smart Energy Forecasting System using Machine Learning
**📌 Project Overview**

The Smart Energy Forecasting System is an end-to-end Machine Learning–based web application that predicts future household energy consumption using historical smart home data. The system leverages time-series modeling (LSTM) to forecast energy usage for the next hour, week, or month and provides device-wise energy estimates along with smart energy-saving suggestions.

This project integrates:

Data preprocessing

Feature engineering

Model training & evaluation

Hyperparameter tuning

REST API development using Flask

Interactive web interface

Visualization dashboards

**🎯 Objectives**

Predict future energy consumption accurately using historical data

Provide device-wise energy consumption insights

Help users optimize energy usage with smart recommendations

Build a complete ML-to-Web deployment pipeline

🧠 Technologies Used
Programming & Libraries

Python 3

NumPy

Pandas

Scikit-learn

TensorFlow / Keras

Matplotlib

Web & Deployment

Flask (REST API)

HTML

CSS

JavaScript

Tools

Jupyter Notebook

VS Code

Git & GitHub

📂 Project Structure
AI_Project/
│
├── data/
│   └── smart_home_energy_complete_dataset.csv
│
├── static/
│   ├── style.css
│   ├── script.js
│   └── plots/
│
├── templates/
│   └── index.html
│
├── app.py
├── best_energy_model.h5
├── scaler_X.pkl
├── scaler_y.pkl
├── README.md
└── requirements.txt

📊 Dataset Description

The dataset contains hourly smart home energy readings with the following features:

Timestamp

Fridge

Wine Cellar

Garage Door

Microwave

Living Room Appliances

Temperature

Humidity

Visibility

Wind Speed

Cloud Cover

Pressure

Total Energy Consumption (Target)

🏗️ Project Modules Summary
**Module 1: Data Understanding & Preprocessing
**
Data cleaning

Timestamp parsing

Handling missing values

Feature scaling using MinMaxScaler

**Module 2: Exploratory Data Analysis**

Trend analysis

Device-wise energy usage patterns

Correlation analysis

**Module 3: Baseline Model**

Linear Regression model

Performance evaluation using MAE, RMSE, R²
**
Module 4: LSTM Model Development**

Time-series sequence creation

LSTM architecture design

Training and validation

**Module 5: Hyperparameter Tuning**

Tuning time steps, units, epochs

Selecting best-performing model
**
Module 6: Model Evaluation & Integration
**
Comparison between baseline and LSTM

Model persistence (.h5, .pkl)

Flask-compatible prediction function

**Module 7: Dashboard & Visualization**

Hourly, daily, weekly, monthly consumption plots

Device-wise energy usage charts

Smart energy-saving suggestions

**Module 8: Web Application Deployment**

Flask API for predictions

Interactive frontend interface

Device selection & prediction horizon

End-to-end ML deployment

🔁 Prediction Workflow

User selects:

Device (e.g., Fridge)

Prediction Horizon (Hour / Week / Month)

System uses last 14 time steps as input

LSTM model predicts future energy consumption

Output displayed:

Total Energy (kWh)

Device-wise Energy (kWh)

Smart Energy Saving Tip

**🌐 Web Application Features**

Interactive UI

Real-time predictions

Device-wise insights

Energy consumption in kWh

Smart energy efficiency tips

Backend powered by Flask API

**🚀 How to Run the Project**
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Flask Server
python app.py

3️⃣ Open Browser
http://127.0.0.1:5000

📈 Sample Output

Next Hour Energy: 0.68 kWh

Fridge Energy Share: 0.14 kWh

Smart Tip: Avoid frequent door opening to reduce cooling loss.

🧪 Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

The tuned LSTM model significantly outperformed the baseline regression model.

**📌 Future Enhancements**

Real-time IoT sensor integration

User authentication & history tracking

Cloud deployment (AWS / Azure)

Mobile-friendly UI

Advanced forecasting (seasonal trends)

**👨‍💻 Author**

Shaik Nasir Ahammed
Computer Science Engineering
Smart Energy Forecasting Project
