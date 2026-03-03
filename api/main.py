# api/main.py

from fastapi import FastAPI
from src.inference import predict
from api.schemas import CustomerData

app = FastAPI(title="Bank Churn Prediction API")

@app.get("/")
def root():
    return {"message": "Bank Churn ML System Running"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    
    result = predict(customer.dict())
    
    return result