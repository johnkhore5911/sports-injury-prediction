import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load scaler and all models
scaler = joblib.load('models/scaler.pkl')
rf = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')
lgb = joblib.load('models/lightgbm.pkl')
models = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgb
}

print("Choose a model:")
for i, name in enumerate(models):
    print(f"{i + 1}. {name}")
model_choice = int(input("Enter model number: ")) - 1
model = list(models.values())[model_choice]

# Collect user input for all features (including BMI)
age = float(input("Enter Player Age: "))
weight = float(input("Enter Player Weight (kg): "))
height = float(input("Enter Player Height (cm): "))
previous_injuries = int(input("Previous Injuries (0 or 1): "))
training_intensity = float(input("Training Intensity (0 to 1): "))
recovery_time = int(input("Recovery Time (days): "))
sleep_hours = float(input("Average Sleep Hours (4-10): "))
hydration_level = float(input("Hydration Level (0 to 1): "))
muscle_fatigue_level = float(input("Muscle Fatigue Level (0 to 1): "))

# Calculate BMI feature
bmi = weight / ((height / 100) ** 2)

# Arrange features in correct order, including BMI as last column
user_features = np.array([[age, weight, height, previous_injuries, training_intensity,
                           recovery_time, sleep_hours, hydration_level, muscle_fatigue_level, bmi]])

# Scale features
user_features_scaled = scaler.transform(user_features)

# Predict injury probability and label
prob = model.predict_proba(user_features_scaled)[0][1]
label = model.predict(user_features_scaled)[0]

print(f"Injury Probability: {prob:.2f}")
print("Prediction:", "Injury" if label == 1 else "No Injury")

# Visualize probability
plt.figure(figsize=(6,1))
plt.barh(['Injury Risk'], [prob], color='red' if prob > 0.5 else 'green')
plt.xlim(0,1)
plt.xlabel('Probability')
plt.title('Injury Prediction Probability')
plt.show()
