from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and scaler
scaler = joblib.load('model/scaler.joblib')
kmeans = joblib.load('model/kmeans_model.joblib')

app = FastAPI(
    title="Mall Customer Segmentation API",
    description="Predict customer segment based on age, income, and spending score.",
    version="1.0"
)

# Example input
template = {
    "Age": 30,
    "Annual_Income": 70,
    "Spending_Score": 50
}

class Customer(BaseModel):
    Age: float
    Annual_Income: float
    Spending_Score: float

# Prediction endpoint
@app.post("/predict")
def predict_segment(customer: Customer):
    data = [[
        customer.Age,
        customer.Annual_Income,
        customer.Spending_Score
    ]]
    df = pd.DataFrame(data, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    X_scaled = scaler.transform(df)
    cluster = int(kmeans.predict(X_scaled)[0])
    return {"cluster": cluster}

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to Mall Customer Segmentation API!",
        "predict_example": template
    }