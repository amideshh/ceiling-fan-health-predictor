
import os
import glob
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from train_model import extract_features

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

def load_folder(folder, use_labels=False):
    files = glob.glob(os.path.join(BASE, folder, "*.wav"))
    features = []
    labels = []

    for path in files:
        name = os.path.basename(path).lower()

        try:
            feature = extract_features(path)
        except Exception as e:
            print("Skipping:", name, e)
            continue

        features.append(feature)

        if use_labels:
            if "_normal_" in name:
                labels.append(0)
            elif "_anomaly_" in name:
                labels.append(1)
            else:
                features.pop()
                continue

    if use_labels:
        return np.asarray(features), np.asarray(labels)

    return np.asarray(features)

print("Loading normal training data...")
X_train = load_folder("train")

print("Loading source test...")
X_source, y_source = load_folder("source_test", True)

print("Loading target test...")
X_target, y_target = load_folder("target_test", True)

if len(X_train) == 0 or len(X_source) == 0 or len(X_target) == 0:
    raise ValueError("A dataset is empty. Check the folder paths.")

print("\nShapes:")
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

print("\nTraining Isolation Forest...")
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)
model.fit(X_train_z)

# Higher score means more anomalous
source_scores = -model.decision_function(X_source_z)
target_scores = -model.decision_function(X_target_z)

# Explore thresholds on source data only.
# Choose the threshold with best source balanced accuracy.
thresholds = np.unique(
    np.percentile(source_scores, np.arange(1, 100))
)

best_threshold = None
best_balanced_accuracy = -1

for threshold in thresholds:
    pred = (source_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_source, pred, labels=[0, 1]
    ).ravel()

    normal_recall = tn / (tn + fp) if (tn + fp) else 0
    anomaly_recall = tp / (tp + fn) if (tp + fn) else 0
    balanced_accuracy = (normal_recall + anomaly_recall) / 2

    if balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = balanced_accuracy
        best_threshold = threshold

print("\nSource-selected threshold:", best_threshold)
print("Source balanced accuracy:", round(best_balanced_accuracy, 4))

# Report source performance at the selected threshold
source_pred = (source_scores >= best_threshold).astype(int)

print("\nSOURCE TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_source, source_pred), 4))
print("Confusion matrix:")
print(confusion_matrix(y_source, source_pred, labels=[0, 1]))
print(classification_report(
    y_source, source_pred,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

# Apply the same threshold unchanged to target data
target_pred = (target_scores >= best_threshold).astype(int)

print("\nTARGET TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_target, target_pred), 4))
print("Confusion matrix:")
print(confusion_matrix(y_target, target_pred, labels=[0, 1]))
print(classification_report(
    y_target, target_pred,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

print("\nTest completed.")
