
import os
import joblib
import numpy as np
import librosa

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

DATASET_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"
TEST_PATH = os.path.join(DATASET_PATH, "target_test")
MODEL_PATH = r"D:\AI_Device_Healt_Prediction\myproject\fan_random_forest.pkl"


# SAME 50 FEATURES USED DURING TRAINING
def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    features = []

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    features.extend([np.mean(rms), np.std(rms)])
    features.extend([np.mean(centroid), np.std(centroid)])
    features.extend([np.mean(bandwidth), np.std(bandwidth)])
    features.extend([np.mean(rolloff), np.std(rolloff)])
    features.extend([np.mean(zcr), np.std(zcr)])

    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    return np.asarray(features, dtype=np.float64)


# Load trained model
model = joblib.load(MODEL_PATH)

X_test = []
y_test = []

if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError("Test folder not found: " + TEST_PATH)

# Read target_test WAV files
for filename in os.listdir(TEST_PATH):
    if not filename.lower().endswith(".wav"):
        continue

    name = filename.lower()

    if "normal" in name:
        label = 0
    elif "anomaly" in name:
        label = 1
    else:
        continue

    path = os.path.join(TEST_PATH, filename)

    try:
        features = extract_features(path)

        if len(features) != 50:
            print("Skipped (wrong feature count):", filename)
            continue

        X_test.append(features)
        y_test.append(label)

    except Exception as error:
        print("Skipped:", filename, error)


X_test = np.asarray(X_test, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.int64)

print("\nTest samples:", len(y_test))
print("Feature shape:", X_test.shape)

if len(y_test) == 0:
    raise ValueError("No test files were processed.")

if X_test.ndim != 2 or X_test.shape[1] != 50:
    raise ValueError("Expected feature shape (number_of_files, 50).")

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
