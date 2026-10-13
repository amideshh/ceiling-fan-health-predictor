
import os
import joblib
import librosa
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset paths
DATASET_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

NORMAL_TRAIN = os.path.join(DATASET_PATH, "train")
SOURCE_TEST = os.path.join(DATASET_PATH, "source_test")
TARGET_TEST = os.path.join(DATASET_PATH, "target_test")

# Save to the Django project
OUTPUT_MODEL = (
    r"D:\AI_Device_Healt_Prediction\myproject\fan_random_forest.pkl"
)


def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13
    )
    rms = librosa.feature.rms(y=audio)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr
    )
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr
    )

    features = []

    # 26 MFCC features
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # RMS, centroid, bandwidth, rolloff and ZCR
    for feature in (rms, centroid, bandwidth, rolloff, zcr):
        features.extend([
            np.mean(feature),
            np.std(feature)
        ])

    # 14 spectral contrast features
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    features = np.asarray(features, dtype=np.float64)

    if features.shape[0] != 50:
        raise ValueError(
            "Expected 50 features, got {}".format(len(features))
        )

    return features


def load_wav_features(folder, label, keyword=None):
    X = []
    y = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            "Dataset folder not found: {}".format(folder)
        )

    files = sorted(os.listdir(folder))

    for filename in files:
        if not filename.lower().endswith(".wav"):
            continue

        if keyword and keyword not in filename.lower():
            continue

        path = os.path.join(folder, filename)

        try:
            features = extract_features(path)
            X.append(features)
            y.append(label)
        except Exception as error:
            print("Skipped:", filename, "|", error)

    print(
        os.path.basename(folder),
        "label =", label,
        "loaded =", len(X)
    )

    return X, y


# 1. Load normal recordings from train
print("\nLoading normal training recordings...")

X_normal, y_normal = load_wav_features(
    NORMAL_TRAIN, 0, keyword="normal"
)

# 2. Load anomaly recordings from source_test
print("\nLoading labelled anomaly recordings...")

X_anomaly, y_anomaly = load_wav_features(
    SOURCE_TEST, 1, keyword="anomaly"
)

X = np.asarray(X_normal + X_anomaly, dtype=np.float64)
y = np.asarray(y_normal + y_anomaly, dtype=np.int64)

print("\nTraining data shape:", X.shape)
print("Normal samples:", int(np.sum(y == 0)))
print("Anomaly samples:", int(np.sum(y == 1)))

if len(X) == 0 or len(np.unique(y)) != 2:
    raise ValueError(
        "Need both normal and anomaly recordings to train."
    )

if X.shape[1] != 50:
    raise ValueError("Training features must contain 50 columns.")

# 3. Hold out part of this training pool for validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_val)

print("\nValidation accuracy:",
      accuracy_score(y_val, predictions))

print("\nValidation report:")
print(classification_report(
    y_val,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

print("Validation confusion matrix:")
print(confusion_matrix(y_val, predictions, labels=[0, 1]))

# 4. Evaluate on the separate target_test domain
print("\nEvaluating on target_test...")

X_target_normal, y_target_normal = load_wav_features(
    TARGET_TEST, 0, keyword="normal"
)

X_target_anomaly, y_target_anomaly = load_wav_features(
    TARGET_TEST, 1, keyword="anomaly"
)

X_target = np.asarray(
    X_target_normal + X_target_anomaly,
    dtype=np.float64
)
y_target = np.asarray(
    y_target_normal + y_target_anomaly,
    dtype=np.int64
)

if len(X_target) > 0 and len(np.unique(y_target)) == 2:
    target_predictions = model.predict(X_target)

    print("\nTarget-test accuracy:",
          accuracy_score(y_target, target_predictions))

    print("\nTarget-test report:")
    print(classification_report(
        y_target,
        target_predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print("Target-test confusion matrix:")
    print(confusion_matrix(
        y_target, target_predictions, labels=[0, 1]
    ))
else:
    print("Target-test evaluation skipped: both classes are required.")

# 5. Retrain on the complete training pool and save
print("\nTraining final model on the training pool...")
model.fit(X, y)

os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

# Save to a temporary file first, then replace the model
TEMP_MODEL = OUTPUT_MODEL + ".new"

joblib.dump(model, TEMP_MODEL)
os.replace(TEMP_MODEL, OUTPUT_MODEL)

print("\nNew model saved successfully:")
print(OUTPUT_MODEL)
print("Feature count:", X.shape[1])
print("scikit-learn version used for training model:")
import sklearn
print(sklearn.__version__)
