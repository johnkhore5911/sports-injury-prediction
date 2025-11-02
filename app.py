from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the scaler and models when the application starts
try:
    scaler = joblib.load('models/scaler.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    xgb_model = joblib.load('models/xgboost.pkl')
    lgb_model = joblib.load('models/lightgbm.pkl')
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure the 'models' directory and all .pkl files are present.")
    # Handle the error appropriately, maybe exit or return an error state
    exit()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        try:
            # --- Get user input from the form ---
            model_name = request.form['model']
            age = float(request.form['age'])
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            previous_injuries = int(request.form['previous_injuries'])
            training_intensity = float(request.form['training_intensity'])
            recovery_time = int(request.form['recovery_time'])
            sleep_hours = float(request.form['sleep_hours'])
            hydration_level = float(request.form['hydration_level'])
            muscle_fatigue_level = float(request.form['muscle_fatigue_level'])

            # --- Preprocessing ---
            # 1. Calculate BMI
            bmi = weight / ((height / 100) ** 2)

            # 2. Create the feature array in the correct order
            feature_names = [
                'PlayerAge', 'PlayerWeight', 'PlayerHeight', 'PreviousInjuries',
                'TrainingIntensity', 'RecoveryTime', 'SleepHours', 'HydrationLevel',
                'MuscleFatigueLevel', 'BMI'
            ]
            user_features = np.array([[
                age, weight, height, previous_injuries, training_intensity,
                recovery_time, sleep_hours, hydration_level, muscle_fatigue_level, bmi
            ]])

            # 3. Scale the features
            user_features_scaled = scaler.transform(user_features)

            # --- Prediction ---
            model = models.get(model_name)
            if model:
                # Predict probability and class
                probability = model.predict_proba(user_features_scaled)[0][1]
                prediction_label = 'Injury' if probability > 0.5 else 'No Injury'

                prediction_result = {
                    "model_name": model_name,
                    "probability": f"{probability:.2%}",
                    "prediction": prediction_label
                }
            else:
                return "Error: Invalid model selected.", 400

        except (ValueError, KeyError) as e:
            # Handle cases where form data is missing or not in the correct format
            return f"Invalid input error: {e}", 400

    return render_template('index.html', result=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
