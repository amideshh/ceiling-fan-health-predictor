#
# import os
#
# base_path = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"
#
# for folder in ["train", "source_test", "target_test"]:
#     path = os.path.join(base_path, folder)
#
#     print("\n", folder)
#
#     if not os.path.isdir(path):
#         print("Folder not found:", path)
#         continue
#
#     files = [
#         name for name in os.listdir(path)
#         if name.lower().endswith(".wav")
#     ]
#
#     print("WAV count:", len(files))
#
#     for name in files[:10]:
#         print(name)

# import joblib
#
# path = r"D:\AI_Device_Healt_Prediction\models\fan_isolation_forest.pkl"
#
# model = joblib.load(path)
#
# print("Model type:", type(model).__name__)
# print("Number of features:", model.n_features_in_)
# print("Normal prediction:", model.predict([[0.0] * 50]))

import os
import glob

base = r"D:\AI_Device_Healt_Prediction\dataset\dev_data_fan\fan"

for folder in ["train", "source_test", "target_test"]:
    path = os.path.join(base, folder)
    files = glob.glob(os.path.join(path, "*.wav"))

    normal = sum("_normal_" in os.path.basename(f).lower()
                 for f in files)
    anomaly = sum("_anomaly_" in os.path.basename(f).lower()
                  for f in files)

    print(folder, "Total:", len(files),
          "Normal:", normal, "Anomaly:", anomaly)
