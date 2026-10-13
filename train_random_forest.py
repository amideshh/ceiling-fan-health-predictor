
import os
import glob
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from train_model import extract_features


DATASET_PATH = (
    r"D:\AI_Device_Healt_Prediction\dataset"
    r"\dev_data_fan\fan\source_test"
)

MODEL_PATH = (
    r"D:\AI_Device_Healt_Prediction\models"
    r"\fan_random_forest_v2.pkl"
)

x = []
y = []

files = glob.glob(os.path.join(DATASET_PATH, "*.wav"))

print("Reading audio files...")

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
            print("Unexpected feature count:", name, len(features))
            continue

        x.append(features)
        y.append(label)

    except Exception as error:
        print("Error:", name, error)

    if (index + 1) % 50 == 0:
        print("Processed:", index + 1, "/", len(files))

x = np.asarray(x, dtype=np.float64)
y = np.asarray(y, dtype=np.int32)

print("\nTotal samples:", len(y))
print("Feature count:", x.shape[1] if len(x) else 0)
print("Normal:", np.sum(y == 0))
print("Anomaly:", np.sum(y == 1))

if len(y) == 0 or len(np.unique(y)) != 2:
    raise ValueError("Dataset must contain normal and anomaly samples.")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining Random Forest...")
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

joblib.dump(model, MODEL_PATH)

print("\nNew model saved:")
print(MODEL_PATH)
