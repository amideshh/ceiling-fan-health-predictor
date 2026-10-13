
import os
import glob
import numpy as np
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"


def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])


def load_folder(folder):
    paths = glob.glob(os.path.join(BASE, folder, "*.wav"))
    features = []
    labels = []

    print("\nReading:", folder)
    print("Files:", len(paths))

    for path in paths:
        name = os.path.basename(path).lower()

        if "_normal_" in name:
            label = 0
        elif "_anomaly_" in name:
            label = 1
        else:
            continue

        try:
            features.append(extract_mfcc(path))
            labels.append(label)
        except Exception as error:
            print("Skipping:", name, error)

    return np.asarray(features), np.asarray(labels)


print("Loading source test...")
X, y = load_folder("source_test")

print("Loading target test...")
X_target, y_target = load_folder("target_test")

if len(X) == 0 or len(X_target) == 0:
    raise ValueError("No features loaded. Check dataset paths.")

print("\nFEATURE SHAPES")
print("Source:", X.shape)
print("Target:", X_target.shape)

# Keep source data split separate for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

def show_results(name, actual, predicted):
    print("\n" + name)
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Balanced accuracy:",
          round(balanced_accuracy_score(actual, predicted), 4))
    print("Confusion matrix [normal, anomaly]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

val_pred = model.predict(X_val)
show_results("SOURCE VALIDATION RESULTS", y_val, val_pred)

# Target set is evaluation only; do not tune from these results
target_pred = model.predict(X_target)
show_results("TARGET EVALUATION RESULTS", y_target, target_pred)

print("\nMFCC Random Forest experiment completed.")
