
import os
import glob
import numpy as np
import librosa

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"


def extract_mfcc(path):
    audio, sr = librosa.load(path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    # 13 MFCC means + 13 MFCC standard deviations
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])


def load_folder(folder, with_labels=False):
    paths = glob.glob(os.path.join(BASE, folder, "*.wav"))
    features = []
    labels = []

    print("\nReading:", folder)
    print("Files:", len(paths))

    for path in paths:
        name = os.path.basename(path).lower()

        try:
            feature = extract_mfcc(path)
        except Exception as error:
            print("Skipping:", name, error)
            continue

        if with_labels:
            if "_normal_" in name:
                label = 0
            elif "_anomaly_" in name:
                label = 1
            else:
                continue

            labels.append(label)

        features.append(feature)

    if with_labels:
        return np.asarray(features), np.asarray(labels)

    return np.asarray(features)


print("Loading normal training data...")
X_train = load_folder("train")

print("\nLoading source test...")
X_source, y_source = load_folder("source_test", True)

print("\nLoading target test...")
X_target, y_target = load_folder("target_test", True)

if len(X_train) == 0 or len(X_source) == 0 or len(X_target) == 0:
    raise ValueError("A folder has no readable audio features.")

print("\nFEATURE SHAPES")
print("Train:", X_train.shape)
print("Source:", X_source.shape)
print("Target:", X_target.shape)

# Standardize using training data only
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std < 1e-8] = 1.0

X_train_z = (X_train - mean) / std
X_source_z = (X_source - mean) / std
X_target_z = (X_target - mean) / std

print("\nTraining MFCC-only Isolation Forest...")
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)
model.fit(X_train_z)

# Larger score means more anomalous
source_scores = -model.decision_function(X_source_z)
target_scores = -model.decision_function(X_target_z)

# Select threshold using source labels only
thresholds = np.unique(
    np.percentile(source_scores, np.arange(1, 100))
)

best_threshold = None
best_score = -1

for threshold in thresholds:
    predictions = (source_scores >= threshold).astype(int)
    score = balanced_accuracy_score(y_source, predictions)

    if score > best_score:
        best_score = score
        best_threshold = threshold

print("\nSource-selected threshold:", best_threshold)
print("Source balanced accuracy:", round(best_score, 4))


def show_results(name, y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    print("\n" + name)
    print("Accuracy:", round(accuracy_score(y_true, predictions), 4))
    print("Balanced accuracy:",
          round(balanced_accuracy_score(y_true, predictions), 4))
    print("Confusion matrix [normal, anomaly]:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))


show_results("SOURCE TEST RESULTS",
             y_source, source_scores, best_threshold)

# Use the exact same threshold on target data
show_results("TARGET TEST RESULTS",
             y_target, target_scores, best_threshold)

print("\nMFCC-only experiment completed.")
