# src/inference.py

import joblib
import pandas as pd
import shap
import numpy as np

from src.config import MODEL_PATH, THRESHOLD


# Load full pipeline
pipeline = joblib.load(MODEL_PATH)

# Extract components
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# Create SHAP explainer using final LightGBM model
explainer = shap.TreeExplainer(model)


def _get_strength(abs_val: float) -> str:
    if abs_val > 0.5:
        return "High"
    elif abs_val > 0.2:
        return "Medium"
    else:
        return "Low"


def predict(data: dict):
    # Raw input dataframe
    df = pd.DataFrame([data])

    # --- Prediction using full pipeline ---
    proba = pipeline.predict_proba(df)[0][1]
    risk = "High" if proba >= THRESHOLD else "Low"

    if risk == "High":
        risk_message = "Customer is above churn intervention threshold. Preventive actions recommended."
    else:
        risk_message = "Customer churn risk is below intervention threshold."

    # --- SHAP Explanation ---
    transformed_df = preprocessor.transform(df)

    shap_values = explainer.shap_values(transformed_df)

    # Handle LightGBM binary output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]

    feature_names = preprocessor.get_feature_names_out()

    # Pair and sort by absolute impact
    sorted_features = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    explanations = []

    for feature_name, shap_value in sorted_features:
        clean_name = feature_name.split("__")[-1]

        # Get original customer value if exists
        customer_value = data.get(clean_name, None)

        direction = (
            "Increases churn risk"
            if shap_value > 0
            else "Decreases churn risk"
        )

        strength = _get_strength(abs(shap_value))

        explanations.append({
            "feature": clean_name,
            "customer_value": customer_value,
            "impact_direction": direction,
            "impact_strength": strength,
            "technical_contribution": float(shap_value)
        })

    return {
        "churn_probability": float(proba),
        "risk_level": risk,
        "threshold": THRESHOLD,
        "risk_interpretation": risk_message,
        "top_drivers": explanations
    }