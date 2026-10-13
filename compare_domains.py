
import os
import glob
import numpy as np

from train_model import extract_features

BASE_PATH = (
    r"D:\AI_Device_Healt_Prediction\dataset"
    r"\dev_data_fan\fan"
)

FOLDERS = ["source_test", "target_test"]

def load_features(folder):
    x_normal = []
    x_anomaly = []

    folder_path = os.path.join(BASE_PATH, folder)
    files = glob.glob(os.path.join(folder_path, "*.wav"))

    print("\nReading:", folder)
    print("Files:", len(files))

    for path in files:
        name = os.path.basename(path).lower()

        if "_normal_" in name:
            target = x_normal
        elif "_anomaly_" in name:
            target = x_anomaly
        else:
            continue

        try:
            target.append(extract_features(path))
        except Exception as error:
            print("Error:", name, error)

    return np.asarray(x_normal), np.asarray(x_anomaly)


data = {}

for folder in FOLDERS:
    normal, anomaly = load_features(folder)

    data[folder] = {
        "normal": normal,
        "anomaly": anomaly
    }

    print("Normal samples:", len(normal))
    print("Anomaly samples:", len(anomaly))


print("\nDOMAIN FEATURE COMPARISON")
print("Feature | Source Normal | Target Normal | Difference")

source_normal = data["source_test"]["normal"]
target_normal = data["target_test"]["normal"]

if len(source_normal) == 0 or len(target_normal) == 0:
    raise ValueError("Normal samples missing from a folder.")

for i in range(source_normal.shape[1]):
    source_mean = np.mean(source_normal[:, i])
    target_mean = np.mean(target_normal[:, i])

    print(
        i + 1,
        round(float(source_mean), 4),
        round(float(target_mean), 4),
        round(float(target_mean - source_mean), 4)
    )


print("\nANOMALY FEATURE COMPARISON")
print("Feature | Source Anomaly | Target Anomaly | Difference")

source_anomaly = data["source_test"]["anomaly"]
target_anomaly = data["target_test"]["anomaly"]

if len(source_anomaly) == 0 or len(target_anomaly) == 0:
    raise ValueError("Anomaly samples missing from a folder.")

for i in range(source_anomaly.shape[1]):
    source_mean = np.mean(source_anomaly[:, i])
    target_mean = np.mean(target_anomaly[:, i])

    print(
        i + 1,
        round(float(source_mean), 4),
        round(float(target_mean), 4),
        round(float(target_mean - source_mean), 4)
    )

print("\nComparison completed.")
