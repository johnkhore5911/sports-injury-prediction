import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

X_test = pd.read_csv('X_test_scaled.csv')
y_test = pd.read_csv('y_test.csv')

# Load models
rf = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')
lgb = joblib.load('models/lightgbm.pkl')

models = {'Random Forest': rf, 'XGBoost': xgb, 'LightGBM': lgb}

metrics = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
    }
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0,1], ['No Injury', 'Injury'])
    plt.yticks([0,1], ['No Injury', 'Injury'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center')
    plt.show()

# Metrics display
results_df = pd.DataFrame(metrics).T
print(results_df)
