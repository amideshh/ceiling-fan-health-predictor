import joblib

model = joblib.load(
    r"D:\AI_Device_Healt_Prediction\models\fan_random_forest.pkl"
)

print("Model classes:", model.classes_)
print("Number of features:", model.n_features_in_)

