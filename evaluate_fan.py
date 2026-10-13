
import os
import glob
import numpy as np
import librosa
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)

BASE_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"
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


def evaluate_folder(model, folder_name):
    folder = os.path.join(BASE_PATH, folder_name)
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))

    X, y = [], []

    print("\nLoading:", folder_name)

    for i, file_path in enumerate(files, 1):
        name = os.path.basename(file_path).lower()

        if "_normal_" in name:
            label = 0
        elif "_anomaly_" in name:
            label = 1
        else:
            continue

        try:
            X.append(extract_features(file_path))
            y.append(label)
        except Exception as error:
            print("Error:", os.path.basename(file_path), error)

        if i % 100 == 0:
            print("Processed:", i, "/", len(files))

    if not X:
        print("No usable audio files found.")
        return

    X = np.asarray(X)
    y = np.asarray(y)

    # IsolationForest: 1 = normal, -1 = anomaly
    raw_predictions = model.predict(X)
    predictions = np.where(raw_predictions == 1, 0, 1)

    print("\nResults:", folder_name)
    print("Samples:", len(y))
    print("Accuracy:", round(accuracy_score(y, predictions) * 100, 2), "%")
    print("Balanced accuracy:",
          round(balanced_accuracy_score(y, predictions) * 100, 2), "%")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y, predictions, labels=[0, 1]))
    print(classification_report(
        y, predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found:", MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)

    evaluate_folder(model, "source_test")
    evaluate_folder(model, "target_test")


if __name__ == "__main__":
    main()
