
import os
import glob
import joblib
import numpy as np
import librosa

from train_model import extract_features

MODEL_PATH = r"D:\AI_Device_Healt_Prediction\models\fan_isolation_forest_v2.pkl"
TEST_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan\source_test"

model = joblib.load(MODEL_PATH)

scores = {"normal": [], "anomaly": []}

for path in glob.glob(os.path.join(TEST_PATH, "*.wav")):
    name = os.path.basename(path).lower()

    if "_normal_" in name:
        label = "normal"
    elif "_anomaly_" in name:
        label = "anomaly"
    else:
        continue

    try:
        features = extract_features(path).reshape(1, -1)
        score = model.decision_function(features)[0]
        scores[label].append(score)
    except Exception as error:
        print("Error:", name, error)

for label, values in scores.items():
    print("\n", label.upper(), "count:", len(values))
    print("Mean score:", round(float(np.mean(values)), 4))
    print("Min score:", round(float(np.min(values)), 4))
    print("Max score:", round(float(np.max(values)), 4))
