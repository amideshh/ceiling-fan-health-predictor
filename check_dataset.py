
import os
import glob
from collections import Counter

BASE_PATH = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

for folder in ["train", "source_test", "target_test"]:
    path = os.path.join(BASE_PATH, folder)
    files = sorted(glob.glob(os.path.join(path, "*.wav")))

    labels = []
    for file in files:
        name = os.path.basename(file).lower()

        if "_normal_" in name:
            labels.append("Normal")
        elif "_anomaly_" in name:
            labels.append("Anomaly")
        else:
            labels.append("Unknown")

    print("\nFolder:", folder)
    print("Total files:", len(files))
    print("Labels:", Counter(labels))
    print("Example:", [os.path.basename(f) for f in files[:3]])
