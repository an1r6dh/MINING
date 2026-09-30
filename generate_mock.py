import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Loading calibrated training samples for hardware model generation...")

# Target Hardware Calibration:
# 1) HC-SR04: Safe (0-40mm), Warning (41-80mm), Danger (>=81mm)
# 2) DFR0028 Tilt: Safe (-2° to +2°), Warning (2.01° to 3.8° / -2.01° to -3.8°), Danger (>3.801° / <= -3.81°)
# 3) DFR0027 Vibration: Safe (0 to 0.20), Warning (0.201 to 0.260), Danger (>=0.261)

if os.path.exists("sensor_data.csv"):
    df = pd.read_csv("sensor_data.csv", nrows=150000)
    X = df[["filtered_tilt", "filtered_vibration", "filtered_strain"]].values.tolist()
    y = df["status"].values.tolist()
else:
    X, y = [], []

# Exact boundary anchor points across all permutation combinations
boundary_anchors = []

# Safe points (strictly within all three safe bounds)
for t in [-1.8, -1.0, -0.5, 0.0, 0.5, 1.0, 1.8]:
    for v in [0.02, 0.05, 0.10, 0.15, 0.18]:
        for d in [5.0, 15.0, 25.0, 35.0, 39.0]:
            boundary_anchors.append(([t, v, d], "SAFE"))

# Warning points: at least one in warning range, none in danger range
for t in [2.1, 2.5, 3.0, 3.5, 3.7, -2.1, -2.5, -3.0, -3.5, -3.7]:
    for v in [0.05, 0.10, 0.15, 0.22, 0.24]:
        for d in [10.0, 20.0, 45.0, 60.0, 75.0]:
            boundary_anchors.append(([t, v, d], "WARNING"))

for t in [-1.5, 0.0, 1.5]:
    for v in [0.205, 0.220, 0.240, 0.255]:
        for d in [10.0, 25.0, 35.0, 50.0, 70.0]:
            boundary_anchors.append(([t, v, d], "WARNING"))

for t in [-1.5, 0.0, 1.5]:
    for v in [0.05, 0.10, 0.15]:
        for d in [42.0, 50.0, 60.0, 70.0, 79.0]:
            boundary_anchors.append(([t, v, d], "WARNING"))

# Danger points: at least one in danger range
for t in [3.85, 4.2, 5.0, 8.0, -3.85, -4.2, -5.0, -8.0]:
    for v in [0.05, 0.15, 0.23, 0.35]:
        for d in [10.0, 30.0, 60.0, 90.0]:
            boundary_anchors.append(([t, v, d], "DANGER"))

for t in [-1.0, 0.0, 1.0, 2.5]:
    for v in [0.265, 0.300, 0.400, 0.800]:
        for d in [10.0, 30.0, 60.0, 90.0]:
            boundary_anchors.append(([t, v, d], "DANGER"))

for t in [-1.0, 0.0, 1.0, 2.5]:
    for v in [0.05, 0.15, 0.23]:
        for d in [82.0, 95.0, 110.0, 140.0]:
            boundary_anchors.append(([t, v, d], "DANGER"))

for pt, lbl in boundary_anchors:
    # Repeat anchors 10 times to give dominant weight to edge boundaries
    for _ in range(10):
        X.append(pt)
        y.append(lbl)

X = np.array(X)
y = np.array(y)

print(f"Fitting RandomForestClassifier on {len(X):,} samples...")
clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=16,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X, y)
joblib.dump(clf, "model.joblib", compress=3)

print("Success: Realistically calibrated 'model.joblib' file generated!")