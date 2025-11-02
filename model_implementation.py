import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# --- ADD THIS BLOCK FOR SMOTE ---
from imblearn.over_sampling import SMOTE

# Load preprocessed data
X_train = pd.read_csv('X_train_scaled.csv')
y_train = pd.read_csv('y_train.csv')

# Apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train.values.ravel())

# --- Random Forest with GridSearchCV ---
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}
rf = RandomForestClassifier(random_state=42)
rf_gs = GridSearchCV(rf, rf_param_grid, scoring='accuracy', cv=3, n_jobs=-1)
rf_gs.fit(X_res, y_res)
print("RandomForest Best Params:", rf_gs.best_params_)
joblib.dump(rf_gs.best_estimator_, 'models/random_forest.pkl')

# --- XGBoost with GridSearchCV ---
xgb_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [6, 10],
    'learning_rate': [0.1, 0.05],
    'subsample': [0.8, 1.0]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_gs = GridSearchCV(xgb, xgb_param_grid, scoring='accuracy', cv=3, n_jobs=-1)
xgb_gs.fit(X_res, y_res)
print("XGBoost Best Params:", xgb_gs.best_params_)
joblib.dump(xgb_gs.best_estimator_, 'models/xgboost.pkl')

# --- LightGBM with GridSearchCV ---
lgb_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [6, 10, -1],
    'learning_rate': [0.1, 0.05],
    'num_leaves': [31, 50]
}
lgb = LGBMClassifier(random_state=42)
lgb_gs = GridSearchCV(lgb, lgb_param_grid, scoring='accuracy', cv=3, n_jobs=-1)
lgb_gs.fit(X_res, y_res)
print("LightGBM Best Params:", lgb_gs.best_params_)
joblib.dump(lgb_gs.best_estimator_, 'models/lightgbm.pkl')

print("All tuned models (SMOTE-balanced) trained and saved.")
