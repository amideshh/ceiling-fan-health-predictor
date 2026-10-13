
import os
import glob
import numpy as np
import librosa
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

BASE_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "source_test")

MODEL_PATH = r"D:\AI_Device_Healt_Prediction\models\fan_isolation_forest_v2.pkl"


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    for values in [rms, centroid, bandwidth, rolloff, zcr]:
        features.append(np.mean(values))
        features.append(np.std(values))

    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    return np.asarray(features, dtype=np.float64)


def load_features(folder):
    X = []
    y = []

    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    print("\nFolder:", folder)
    print("Total WAV files:", len(files))

    for i, path in enumerate(files, 1):
        try:
            X.append(extract_features(path))

            name = os.path.basename(path).lower()
            if "_normal_" in name:
                y.append(0)
            elif "_anomaly_" in name:
                y.append(1)
            else:
                y.append(-1)

        except Exception as error:
            print("Error:", os.path.basename(path), error)

        if i % 100 == 0:
            print("Processed:", i, "/", len(files))

    return np.asarray(X), np.asarray(y)


def main():
    X_train, _ = load_features(TRAIN_PATH)

    if len(X_train) == 0:
        print("ERROR: No training audio files found.")
        return

    print("\nTraining samples:", len(X_train))
    print("Feature count:", X_train.shape[1])

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")
    model.fit(X_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved:", MODEL_PATH)

    X_test, y_test = load_features(TEST_PATH)
    valid = y_test >= 0
    X_test = X_test[valid]
    y_test = y_test[valid]

    if len(y_test) == 0 or len(np.unique(y_test)) < 2:
        print("ERROR: Test data must contain normal and anomaly files.")
        return

    raw_predictions = model.predict(X_test)
    predictions = np.where(raw_predictions == 1, 0, 1)

    print("\nConfusion Matrix (rows=true, columns=predicted)")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification Report")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))


if __name__ == "__main__":
    main()
