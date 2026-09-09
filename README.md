# 🚨 Mine Hazard & Subsidence Early Warning System
**Smart India Hackathon (SIH) Project 26025**

An early warning system that analyzes IoT sensor telemetry (Tilt, Vibration, and Strain) using Machine Learning to detect structural hazards and mine subsidence risks (`SAFE`, `WARNING`, `DANGER`) in real time.

---

## 🔑 Authentication & Admin Credentials

The Streamlit dashboard is protected by an authentication screen with **Sign In** and **Register** tabs. User credentials are automatically persisted in `users.json`.

### Secret Administrator Access:
* **Username**: `Admin`
* **Password**: `godisgreat`
* **Role**: `Administrator`

*New operators, engineers, and inspectors can create custom accounts via the **Register** tab.*

---

## 🏗️ Architecture Components

1. **`main.py`** — FastAPI REST API serving machine learning risk predictions at `POST /predict`.
2. **`dashboard.py`** — Interactive Streamlit real-time monitoring dashboard with Sign In/Register auth, live sensor trend charts, alert summary cards, and multi-scenario simulations.
3. **`simulate_sensor.py`** — CLI IoT sensor node stream simulator.
4. **`train_model.py`** — Model trainer using Random Forest Classifier (auto-generates `sensor_data.csv` if missing).
5. **`generate_mock.py`** — Lightweight decision-tree mock model generator.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic streamlit pandas scikit-learn joblib requests
```

### 2. Generate Model or Train Classifier
To generate a mock model artifact:
```bash
python generate_mock.py
```
Or to train a full Random Forest model (auto-creates synthetic training dataset if needed):
```bash
python train_model.py
```

### 3. Start FastAPI Backend Server
```bash
uvicorn main:app --reload --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- API Documentation: `http://127.0.0.1:8000/docs`

### 4. Launch Telemetry Dashboard (Streamlit)
```bash
streamlit run dashboard.py
```

### 5. (Optional) Run Independent IoT Sensor Simulator
```bash
python simulate_sensor.py
```
