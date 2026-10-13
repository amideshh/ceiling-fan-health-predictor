
import os
import glob
from collections import Counter

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan\train"

files = glob.glob(os.path.join(BASE, "*.wav"))

for section in ["00", "01", "02"]:
    selected = [
        os.path.basename(f).lower()
        for f in files
        if "section_" + section + "_" in os.path.basename(f).lower()
    ]

    print("\nSection:", section)
    print("Total:", len(selected))
    print("Source:", sum("_source_train_" in f for f in selected))
    print("Target:", sum("_target_train_" in f for f in selected))
    print("Labels:", Counter(
        "Normal" if "_normal_" in f else
        "Anomaly" if "_anomaly_" in f else "Unknown"
        for f in selected
    ))
