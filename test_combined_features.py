
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


def extract_features(path):
    audio, sr = librosa.load(path, sr=None, mono=True)

    # MFCC features: 13 means + 13 standard deviations
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13
    )

    features = [
        *np.mean(mfcc, axis=1),
        *np.std(mfcc, axis=1)
    ]

    # Spectral features
    rms = librosa.feature.rms(y=audio)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr
    )[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

    for values in [rms, centroid, bandwidth, rolloff, zcr]:
        features.append(np.mean(values))
        features.append(np.std(values))

    # Spectral contrast: 7 means + 7 standard deviations
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr
    )

    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    return np.asarray(features, dtype=float)


def load_folder(folder):
    paths = glob.glob(os.path.join(BASE, folder, "*.wav"))
    X = []
    y = []

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
            X.append(extract_features(path))
            y.append(label)
        except Exception as error:
            print("Skipping:", name, error)

    return np.asarray(X), np.asarray(y)


print("Loading source test...")
X, y = load_folder("source_test")

print("\nLoading target test...")
X_target, y_target = load_folder("target_test")

if len(X) == 0 or len(X_target) == 0:
    raise ValueError("No features loaded. Check dataset paths.")

print("\nFEATURE SHAPES")
print("Source:", X.shape)
print("Target:", X_target.shape)

# Split source data into training and validation sets
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

print("\nTraining combined-feature Random Forest...")
model.fit(X_train, y_train)


def show_results(name, actual, predicted):
    print("\n" + name)
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Balanced accuracy:",
          round(balanced_accuracy_score(actual, predicted), 4))
    print("Confusion matrix [normal, anomaly]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print(classification_report(
        actual, predicted,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))


val_pred = model.predict(X_val)
show_results("SOURCE VALIDATION RESULTS", y_val, val_pred)

# Target data is evaluation only; do not tune using its labels
target_pred = model.predict(X_target)
show_results("TARGET EVALUATION RESULTS", y_target, target_pred)

print("\nCombined-feature experiment completed.")
