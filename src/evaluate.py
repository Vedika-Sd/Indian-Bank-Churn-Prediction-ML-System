# src/evaluate.py

import pandas as pd
import json
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

from config import TEST_PATH, MODEL_PATH, METRICS_PATH

def main():

    test = pd.read_csv(TEST_PATH)

    X_test = test.drop(columns=["Churn", "CustomerID"])
    y_test = test["Churn"]

    model = joblib.load(MODEL_PATH)

    y_proba = model.predict_proba(X_test)[:,1]

    roc = roc_auc_score(y_test, y_proba)
    pr  = average_precision_score(y_test, y_proba)

    metrics = {
        "roc_auc": roc,
        "pr_auc": pr
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete. Metrics saved.")

if __name__ == "__main__":
    main()