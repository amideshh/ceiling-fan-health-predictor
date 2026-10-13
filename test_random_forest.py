
import os
import glob
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from train_model import extract_features

MODEL_PATH = (
    r"D:\AI_Device_Healt_Prediction\models"
    r"\fan_random_forest_v2.pkl"
)

TEST_PATH = (
    r"D:\AI_Device_Healt_Prediction\dataset"
    r"\dev_data_fan\fan\target_test"
)

model = joblib.load(MODEL_PATH)

x_test = []
y_test = []

files = glob.glob(os.path.join(TEST_PATH, "*.wav"))

print("Testing on target_test...")
print("Total audio files:", len(files))

for index, path in enumerate(files):
    name = os.path.basename(path).lower()

    if "_normal_" in name:
        label = 0
    elif "_anomaly_" in name:
        label = 1
    else:
        continue

    try:
        features = extract_features(path)

        if len(features) != 50:
            print("Unexpected feature count:", name)
            continue

        x_test.append(features)
        y_test.append(label)

    except Exception as error:
        print("Error:", name, error)

    if (index + 1) % 50 == 0:
        print("Processed:", index + 1, "/", len(files))

x_test = np.asarray(x_test, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.int32)

if len(y_test) == 0:
    raise ValueError("No test samples loaded.")

predictions = model.predict(x_test)

print("\nTest samples:", len(y_test))
print("Normal:", np.sum(y_test == 0))
print("Anomaly:", np.sum(y_test == 1))

print("\nIndependent Test Accuracy:",
      round(accuracy_score(y_test, predictions), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))
