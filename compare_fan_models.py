
import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)

DATASET_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
SOURCE_TEST_PATH = os.path.join(DATASET_PATH, "source_test")


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


X = []
y = []


def load_folder(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".wav"):
            continue

        name = filename.lower()

        if "normal" in name:
            label = 0
        elif "anomaly" in name:
            label = 1
        else:
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            features = extract_features(file_path)

            if len(features) == 50:
                X.append(features)
                y.append(label)
            else:
                print("Wrong feature count:", filename)

        except Exception as error:
            print("Skipped:", filename, error)


print("Loading normal training recordings...")
load_folder(TRAIN_PATH)

print("Loading labelled source_test recordings...")
load_folder(SOURCE_TEST_PATH)

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.int64)

print("\nDataset shape:", X.shape)
print("Normal recordings:", np.sum(y == 0))
print("Anomaly recordings:", np.sum(y == 1))

if len(y) == 0 or X.ndim != 2 or X.shape[1] != 50:
    raise ValueError("Dataset loading failed or feature count is incorrect.")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale"
        )
    )
}

results = []

for name, model in models.items():
    print("\n" + "=" * 45)
    print("Training:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, pos_label=1, zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Anomaly precision:", round(precision, 4))
    print("Anomaly recall:", round(recall, 4))
    print("Anomaly F1:", round(f1, 4))

    print("\nClassification report:")
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    results.append((name, precision, recall, f1, accuracy))

print("\n\nMODEL COMPARISON")
print("-" * 65)
print("{:<18} {:>10} {:>10} {:>10} {:>10}".format(
    "Model", "Precision", "Recall", "F1", "Accuracy"
))

for name, precision, recall, f1, accuracy in sorted(
    results, key=lambda item: item[3], reverse=True
):
    print("{:<18} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
        name, precision, recall, f1, accuracy
    ))
