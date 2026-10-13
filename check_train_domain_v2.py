
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

    return np.asarray(values, dtype=float)

train = get_features("train")
source = get_features("source_test")
target = get_features("target_test")

if len(train) == 0 or len(source) == 0 or len(target) == 0:
    raise ValueError("One or more folders contain no readable audio.")

# Calculate scale from training data only
mean = np.mean(train, axis=0)
std = np.std(train, axis=0)
std[std < 1e-8] = 1.0

train_z = (train - mean) / std
source_z = (source - mean) / std
target_z = (target - mean) / std

print("STANDARDIZED FEATURE COMPARISON")
print("Train normal baseline: 0.0000")

print("Source test average absolute standardized difference:",
      round(float(np.mean(np.abs(np.mean(source_z, axis=0)))), 4))

print("Target test average absolute standardized difference:",
      round(float(np.mean(np.abs(np.mean(target_z, axis=0)))), 4))

print("Completed.")
