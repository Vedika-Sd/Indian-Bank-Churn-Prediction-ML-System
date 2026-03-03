# src/train.py

import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from config import TRAIN_PATH, MODEL_PATH, MODEL_PARAMS
from preprocessing import build_preprocessor

def main():

    train = pd.read_csv(TRAIN_PATH)

    X = train.drop(columns=["Churn", "CustomerID"])
    y = train["Churn"]

    preprocessor = build_preprocessor(X)

    model = LGBMClassifier(**MODEL_PARAMS)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()