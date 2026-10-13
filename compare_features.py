
import os
import glob
import numpy as np

from train_model import extract_features

DATASET_PATH = (
    r"D:\AI_Device_Healt_Prediction\dataset"
    r"\dev_data_fan\fan\source_test"
)

normal_features = []
anomaly_features = []

for path in glob.glob(os.path.join(DATASET_PATH, "*.wav")):
    name = os.path.basename(path).lower()

    if "_normal_" in name:
        target = normal_features
    elif "_anomaly_" in name:
        target = anomaly_features
    else:
        continue

    try:
        target.append(extract_features(path))
    except Exception as error:
        print("Error:", name, error)

normal = np.asarray(normal_features)
anomaly = np.asarray(anomaly_features)

print("Normal samples:", len(normal))
print("Anomaly samples:", len(anomaly))

print("\nFeature comparison:")
print("Feature | Normal mean | Anomaly mean | Difference")

for i in range(normal.shape[1]):
    normal_mean = np.mean(normal[:, i])
    anomaly_mean = np.mean(anomaly[:, i])

    print(
        i + 1,
        round(float(normal_mean), 4),
        round(float(anomaly_mean), 4),
        round(float(anomaly_mean - normal_mean), 4)
    )
