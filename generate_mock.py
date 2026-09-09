import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1. Define sample inputs: [tilt, vibration, strain]
# Representing SAFE, WARNING, and DANGER telemetry patterns
X = np.array([
    [0.01, 0.05, 0.01],
    [0.05, 0.15, 0.03],
    [0.20, 0.35, 0.10],
    [1.50, 0.65, 0.80],
    [2.50, 0.85, 1.20],
    [3.20, 1.10, 1.80],
    [5.50, 1.80, 3.20],
    [8.00, 2.50, 4.50],
    [12.0, 4.00, 6.00],
])

# 2. Define matching status labels
y = np.array([
    "SAFE", "SAFE", "SAFE",
    "WARNING", "WARNING", "WARNING",
    "DANGER", "DANGER", "DANGER"
])

# 3. Fit a decision tree classifier and save model file
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)
joblib.dump(clf, "model.joblib")

print("Success: Realistically calibrated 'model.joblib' file generated!")