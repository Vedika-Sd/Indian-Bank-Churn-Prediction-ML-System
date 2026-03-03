# Indian Bank Churn Prediction ML System

End-to-end churn prediction system for Indian retail banking, built as a reproducible ML pipeline from EDA and experimentation notebooks to production inference via FastAPI.

## 1. Business Objective

Customer churn is costly for banks because acquisition cost is typically much higher than retention cost. This project predicts churn risk at customer level so retention teams can prioritize interventions.

Primary decision goal:
- Rank and flag customers by churn probability.
- Use thresholded risk labels (`High` / `Low`) for operational action.
- Support interpretation with feature-level contribution signals.

## 2. System Thinking and Architecture

This repository is structured as a full ML system, not only a single model notebook:

1. Data acquisition and understanding (`notebooks/01_eda.ipynb`)
2. Model experimentation (`notebooks/02_experimentation.ipynb`)
3. Feature engineering checks (`notebooks/03_feature_engineering.ipynb`)
4. Hyperparameter tuning (`notebooks/04_hyperparameter_tuning.ipynb`)
5. Final evaluation and thresholding (`notebooks/05_evaluation.ipynb`)
6. Production pipeline training (`src/train.py`)
7. Evaluation artifact generation (`src/evaluate.py` -> `reports/metrics.json`)
8. Inference service with explainability (`src/inference.py`, `api/main.py`)
9. Containerized serving (`Dockerfile`)

## 🚀 Live API

Base URL:
https://indian-bank-churn-prediction-ml-system.onrender.com

Swagger Docs:
https://indian-bank-churn-prediction-ml-system.onrender.com/docs
<br>
<img width="1366" height="653" alt="image" src="https://github.com/user-attachments/assets/7d186513-698f-4d7b-ae0c-680d8dff61a0" />


High-level runtime flow:
- Input JSON -> FastAPI (`/predict`) -> `src.inference.predict()`
- Inference uses persisted sklearn `Pipeline` (`models/model.pkl`)
- Pipeline handles preprocessing + LightGBM prediction
- SHAP computes top contributing transformed features
- API returns probability, risk level, threshold, and top feature contributions

## 3. Data Work

## Dataset Snapshot

- Raw data: `data/raw/Indian_bank_churn_orig.csv`
- Raw shape: `50,000 x 20`
- Train shape: `40,000 x 20`
- Test shape: `10,000 x 20`
- Churn rate (raw): `17.452%`
- Churn rate (train/test): `~17.45%` (stable split)
- Overall raw missingness: `~5.83%`
- Cities: `40` unique Indian cities

Target:
- `Churn` (binary: 1 = churn, 0 = non-churn)

Core feature groups:
- Demographics: `Age`, `Gender`, `City`, `Employment_Type`
- Financial: `Annual_Income`, `Credit_Score`, `Avg_Monthly_Balance`, `Balance_Change_Ratio`
- Behavioral usage: app/netbanking/UPI/ATM usage columns
- Service quality signals: `Complaint_Tickets`, `Call_Center_Interactions`
- Product holding: `Has_Credit_Card`, `Has_Loan`, `Has_Insurance`

## Data and Missing-Value Strategy

Production preprocessing (`src/preprocessing.py`):
- Numeric columns:
  - `SimpleImputer(strategy="median", add_indicator=True)`
- Categorical columns:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`
- Composed via `ColumnTransformer`, then chained with model in one sklearn `Pipeline`.

Why this is system-safe:
- Training/inference preprocessing consistency is guaranteed.
- Missingness indicators preserve information from null patterns.
- Unknown categories at serving time are handled gracefully.

## 4. ML Work: What Was Tried and How

Notebook progression shows structured model development:

### `notebooks/02_experimentation.ipynb`
Initial model comparison (ROC-AUC):
1. Baseline Logistic Regression: `0.7728`
2. Logistic Regression + class weight: `0.7724`
3. XGBoost + `scale_pos_weight`: `0.7743`
4. Random Forest + class weight: `0.7627`
5. LightGBM + class weight: `0.7817`

Baseline selected for next phase: LightGBM-based pipeline (`~0.7802` with refined preprocessing).

### `notebooks/03_feature_engineering.ipynb`
Feature engineering variants were tested, but generalization did not improve beyond the established baseline (`Test ROC-AUC ~0.7802`), indicating added features mostly increased training fit without meaningful test lift.

### `notebooks/04_hyperparameter_tuning.ipynb`
Manual structured tuning to reduce overfitting and improve test ROC-AUC.
Final selected configuration:
- `class_weight="balanced"`
- `num_leaves=15`
- `max_depth=12`
- `min_child_samples=100`
- `random_state=42`

Observed tuned result in notebooks:
- Train ROC-AUC: `0.8109`
- Test ROC-AUC: `0.7866`

### `notebooks/05_evaluation.ipynb`
Final evaluation and threshold analysis:
- Train ROC-AUC: `0.8109`
- Test ROC-AUC: `0.7866`
- PR-AUC (Average Precision): `0.4988`
- Confusion Matrix @ threshold `0.5`: `[[6062, 2193], [545, 1200]]`
- Best F1 threshold found: `0.614062...` (implemented as `0.61` in config)

Top-k targeting analysis from notebook:
- Top 5%: Recall `0.2034`, Precision `0.7100`, Lift `4.07`
- Top 10%: Recall `0.3381`, Precision `0.5900`, Lift `3.38`
- Top 20%: Recall `0.5301`, Precision `0.4625`, Lift `2.65`

## 5. Production Model and Metrics

Current production model artifact:
- File: `models/model.pkl`
- Type: `Pipeline(preprocessor + LGBMClassifier)`
- Model class: `LGBMClassifier`

Config (`src/config.py`):
- Hyperparameters match notebook-selected values
- Decision threshold: `0.61`

Stored evaluation artifact (`reports/metrics.json`):
- `roc_auc`: `0.7865557559107184`
- `pr_auc`: `0.4988369174068161`

## 6. Explainability and Inference Design

Implemented in `src/inference.py`:
- Loads full pipeline once
- Generates churn probability using identical training-time preprocessing
- Applies business threshold (`THRESHOLD=0.61`) to emit risk level
- Uses `shap.TreeExplainer` on LightGBM model
- Returns top 5 transformed feature contributions

Prediction response shape:
- `churn_probability`
- `risk_level`
- `threshold`
- `top_features`

## 7. API Layer

FastAPI app (`api/main.py`):
- `GET /` health-style root message
- `POST /predict` for churn inference

Request schema (`api/schemas.py`) includes all operational input fields used by the model pipeline.

### Example request

```json
{
  "Age": 35,
  "Gender": "Male",
  "City": "Mumbai",
  "Employment_Type": "Salaried",
  "Annual_Income": 750000,
  "Credit_Score": 720,
  "Tenure_Months": 24,
  "Avg_Monthly_Balance": 50000,
  "Balance_Change_Ratio": 0.05,
  "Mobile_App_Logins": 15,
  "UPI_Transactions": 20,
  "ATM_Withdrawals": 3,
  "NetBanking_Usage": 10,
  "Complaint_Tickets": 0,
  "Call_Center_Interactions": 1,
  "Has_Credit_Card": 1,
  "Has_Loan": 0,
  "Has_Insurance": 1
}
```

## 8. Project Structure

```
indian-bank-churn-ml-system/
├── data/
│   ├── raw/
│   │   └── Indian_banking_churn.csv
│   ├── processed/
│   │   └── test.csv
│   │   └── train.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_experimentation.ipynb
│   ├── 03_feature_enginerring.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_evaluation.ipynb
|
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│
├── models/
│   ├── model.pkl (contains preprocessing + model)
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── reports/
│   ├── metrics.json
│
├── Dockerfile
├── requirements.txt
└── [README.md]

```

## 9. Reproducible Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python src/train.py
```

Evaluate model:

```bash
python src/evaluate.py
```

Run API locally:

```bash
python -m uvicorn api.main:app --reload
```

## 10. Docker Deployment

Build image:

```bash
docker build -t bank-churn-api .
```

Run container:

```bash
docker run -p 8000:8000 bank-churn-api
```

Service endpoint:
- `http://localhost:8000/docs`

## 11. Current Strengths and Next Maturity Steps

Current strengths:
- Clean end-to-end pipeline from notebook research to API deployment
- Single serialized pipeline avoids training-serving skew
- Threshold tuning included for business actionability
- SHAP-based local explanations integrated into inference

Next engineering upgrades (recommended):
1. Add CI checks (lint, import test, smoke inference test)
2. Add model/data drift monitoring simulation and alert thresholds
3. Track experiments and artifacts with MLflow/DVC
4. Add calibration checks (Brier score, reliability curve)
5. Add API auth, request logging, and model version endpoint


