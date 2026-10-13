
import os
import glob
import numpy as np
from train_model import extract_features

BASE = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

def get_features(folder):
    files = glob.glob(os.path.join(BASE, folder, "*.wav"))
    values = []

    for file in files:
        try:
            values.append(extract_features(file))
        except Exception as e:
            print("Error:", os.path.basename(file), e)

    return np.asarray(values)

train = get_features("train")
source = get_features("source_test")
target = get_features("target_test")

print("\nFEATURE MATRIX SHAPES")
print("Train:", train.shape)
print("Source test:", source.shape)
print("Target test:", target.shape)

print("\nMEAN ABSOLUTE DIFFERENCE FROM TRAIN")
for name, data in [("Source", source), ("Target", target)]:
    if len(train) and len(data):
        difference = np.abs(np.mean(data, axis=0) -
                            np.mean(train, axis=0))
        print(name, "average feature difference:",
              round(float(np.mean(difference)), 4))

print("\nCompleted.")
