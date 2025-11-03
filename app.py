from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the machine learning models and the scaler
# These are loaded once when the application starts for efficiency.
scaler = joblib.load('models/scaler.pkl')
rf_model = joblib.load('models/random_forest.pkl')
xgb_model = joblib.load('models/xgboost.pkl')
lgb_model = joblib.load('models/lightgbm.pkl')

# Dictionary to hold the models for easy selection
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model
}

@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    try:
        # Get all the input values from the form
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        previous_injuries = int(request.form['previous_injuries'])
        training_intensity = float(request.form['training_intensity'])
        recovery_time = int(request.form['recovery_time'])
        sleep_hours = float(request.form['sleep_hours'])
        hydration_level = float(request.form['hydration_level'])
        muscle_fatigue = float(request.form['muscle_fatigue'])
        model_name = request.form['model']

        # Calculate BMI from weight and height
        bmi = weight / ((height / 100) ** 2)

        # Create a numpy array with the features in the correct order for the model
        features = np.array([[
            age, weight, height, previous_injuries, training_intensity,
            recovery_time, sleep_hours, hydration_level, muscle_fatigue, bmi
        ]])

        # Scale the features using the pre-fitted scaler
        scaled_features = scaler.transform(features)

        # Select the chosen model
        model = models[model_name]

        # Make the prediction and get the probability
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        # Format the output strings
        prediction_text = 'Injury is Likely' if prediction == 1 else 'Injury is Unlikely'
        probability_text = f'Injury Probability: {probability:.2%}'

    except Exception as e:
        # Handle errors gracefully
        prediction_text = f'Error: {e}'
        probability_text = 'Please check your inputs.'

    # Render the page again with the prediction results
    return render_template('index.html', prediction_text=prediction_text, probability_text=probability_text)

if __name__ == '__main__':
    # Run the Flask app with multi-threading enabled
    app.run(debug=True, threaded=True)
