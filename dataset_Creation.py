import numpy as np
import pandas as pd

np.random.seed(42)
num_samples = 5000

ages = np.random.randint(18, 40, size=num_samples)
weights = np.random.normal(70, 10, size=num_samples).clip(50, 100)
heights = np.random.normal(170, 10, size=num_samples).clip(150, 200)
prev_injuries = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
training_intensity = np.random.beta(a=2, b=5, size=num_samples)
recovery_time = np.random.poisson(lam=5, size=num_samples)
sleep_hours = np.random.normal(7, 1, size=num_samples).clip(4, 10)
hydration_level = np.random.beta(a=4, b=2, size=num_samples)
muscle_fatigue_level = np.random.beta(a=5, b=2, size=num_samples)
bmi = weights / ((heights / 100) ** 2)

# -- Maximize predictability without being fully deterministic --
score = (
    2.7 * training_intensity +
    2.5 * prev_injuries +
    2.2 * muscle_fatigue_level +
    0.8 * (10 - recovery_time) / 10 +
    0.7 * (10 - sleep_hours) / 10 +
    0.6 * (1 - hydration_level) +
    1.1 * (bmi > 27.5) +
    0.9 * ((ages > 32) | (ages < 21))
)

prob_injury = 1 / (1 + np.exp(-score))
# Very little label noise for high accuracy
flip = np.random.binomial(1, 0.01, size=num_samples)
injury_outcome = np.where(flip, 1 - (prob_injury > 0.48).astype(int), (prob_injury > 0.48).astype(int))

df = pd.DataFrame({
    'Player_Age': ages,
    'Player_Weight': weights,
    'Player_Height': heights,
    'Previous_Injuries': prev_injuries,
    'Training_Intensity': training_intensity,
    'Recovery_Time': recovery_time,
    'Sleep_Hours': sleep_hours,
    'Hydration_Level': hydration_level,
    'Muscle_Fatigue_Level': muscle_fatigue_level,
    'BMI': bmi,
    'Likelihood_of_Injury': injury_outcome
})

df.to_csv('synthetic_injury_dataset_with_BMI.csv', index=False)
print('Optimized and easily predictable synthetic injury dataset created and saved as synthetic_injury_dataset_with_BMI.csv')
