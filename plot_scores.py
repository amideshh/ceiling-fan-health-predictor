
import os
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt

from train_model import extract_features

MODEL_PATH = r"D:\AI_Device_Healt_Prediction\models\fan_isolation_forest_v2.pkl"
TEST_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan\source_test"

model = joblib.load(MODEL_PATH)

normal_scores = []
anomaly_scores = []

for path in glob.glob(os.path.join(TEST_PATH, "*.wav")):
    name = os.path.basename(path).lower()

    if "_normal_" in name:
        label = "normal"
    elif "_anomaly_" in name:
        label = "anomaly"
    else:
        continue

    features = extract_features(path).reshape(1, -1)
    score = model.decision_function(features)[0]

    if label == "normal":
        normal_scores.append(score)
    else:
        anomaly_scores.append(score)

plt.hist(normal_scores, bins=30, alpha=0.6, label="Normal")
plt.hist(anomaly_scores, bins=30, alpha=0.6, label="Anomaly")

plt.xlabel("Isolation Forest Score")
plt.ylabel("Number of Audio Samples")
plt.title("Fan Normal vs Anomaly Score Distribution")
plt.legend()
plt.tight_layout()
plt.show()
