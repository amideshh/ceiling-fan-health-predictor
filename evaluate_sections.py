
import os
import glob
import numpy as np
import librosa
import joblib

from sklearn.metrics import accuracy_score, balanced_accuracy_score

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"
MODEL = r"D:\AI_Device_Healt_Prediction\models\fan_isolation_forest_v2.pkl"


def features(path):
    audio, sr = librosa.load(path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    result = list(np.mean(mfcc, axis=1))
    result += list(np.std(mfcc, axis=1))

    for values in [rms, centroid, bandwidth, rolloff, zcr]:
        result.extend([np.mean(values), np.std(values)])

    result += list(np.mean(contrast, axis=1))
    result += list(np.std(contrast, axis=1))

    return result


model = joblib.load(MODEL)

for folder_name in ["source_test", "target_test"]:
    print("\n======", folder_name, "======")

    folder = os.path.join(BASE, folder_name)

    for section in ["00", "01", "02"]:
        X, y = [], []

        for path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
            name = os.path.basename(path).lower()

            if "section_" + section + "_" not in name:
                continue

            if "_normal_" in name:
                label = 0
            elif "_anomaly_" in name:
                label = 1
            else:
                continue

            try:
                X.append(features(path))
                y.append(label)
            except Exception as e:
                print("File error:", name, e)

        if not X or len(set(y)) < 2:
            print("Section", section, "- insufficient data")
            continue

        raw = model.predict(np.asarray(X))
        pred = np.where(raw == 1, 0, 1)

        print(
            "Section", section,
            "| Accuracy:", round(accuracy_score(y, pred) * 100, 2), "%",
            "| Balanced:", round(
                balanced_accuracy_score(y, pred) * 100, 2
            ), "%"
        )
