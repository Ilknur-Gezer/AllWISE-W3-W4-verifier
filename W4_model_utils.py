# model_utils.py

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import W4_config

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def train_models():
    df = pd.read_csv(W4_config.TRAINING_DATA)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    models = {
        "rfc": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0, class_weight="balanced"),
        "svc": SVC(kernel="rbf", gamma="auto", probability=True, random_state=0),
        "lr": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=10),
        "nb": GaussianNB(),
        "dt": DecisionTreeClassifier(criterion="entropy", random_state=0)
    }

    best_model_key = None
    best_accuracy = 0.0

    for name, model in models.items():
        accuracies = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        print(f"{name.upper()} Accuracy: {avg_acc:.4f}")

        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_model_key = name

    # Save the best model
    os.makedirs(W4_config.MODEL_DIR, exist_ok=True)
    best_model = models[best_model_key]
    best_model_path = os.path.join(W4_config.MODEL_DIR, f"model_{best_model_key}.pkl")
    joblib.dump(best_model, best_model_path)

    # Save the path to best model
    with open(W4_config.BEST_MODEL_PATH_FILE, "w") as f:
        f.write(best_model_path)

    print(f"Best model is {best_model_key.upper()} with accuracy {best_accuracy:.4f}")
    print(f"Model saved to: {best_model_path}")

def predict_data():
    # Load best model path
    with open(W4_config.BEST_MODEL_PATH_FILE, "r") as f:
        best_model_path = f.read().strip()
    model = joblib.load(best_model_path)

    df = pd.read_csv(W4_config.PREDICTION_DATA)
    pred_cols = df.columns[1:]  # assumes first column is ID
    y_pred = model.predict(df[pred_cols])
    y_prob = model.predict_proba(df[pred_cols])[:, 1]

    pred_class = pd.Series(y_pred).copy()
    pred_class[y_prob <= W4_config.THRESHOLD] = 'fake'
    pred_class[y_prob > W4_config.THRESHOLD] = 'real'

    result = pd.concat([df.iloc[:, 0], pred_class, pd.Series(y_prob)], axis=1)
    result.columns = ['AllWISE', 'class_W4', 'Prob_W4_real']
    result = result.round(4)
    result.to_csv("/Users/igezer/Desktop/allwise_cleaning/AllWISE_W4/AllWISE_OrionFOV_W4_classification_result.csv", index=False)
    print("Predictions saved to: AllWISE_OrionFOV_W4_classification_result.csv")
