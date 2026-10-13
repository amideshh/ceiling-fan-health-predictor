
import os
import glob
from collections import Counter

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

for folder in ["train", "source_test", "target_test"]:
    files = glob.glob(os.path.join(BASE, folder, "*.wav"))
    sections = Counter()

    for file in files:
        name = os.path.basename(file)
        section = name.split("_")[1]
        sections[section] += 1

    print("\n", folder)
    print(sections)
