import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================
# Load Trained Model & Threshold
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "..", "models", "decision_threshold.pkl")

model = joblib.load(MODEL_PATH)

# Load the tuned decision threshold if available; fallback to 0.5
if os.path.exists(THRESHOLD_PATH):
    decision_threshold = joblib.load(THRESHOLD_PATH)
    print(f"Loaded tuned decision threshold: {decision_threshold:.3f}")
else:
    decision_threshold = 0.5
    print("Tuned threshold not found. Using default 0.5")

# ==========================================
# Create FastAPI App
# ==========================================

app = FastAPI(
    title="Customer Purchase Prediction API",
    description="Predict whether a customer is likely to purchase based on demographic, behavioural, and campaign history data.",
    version="2.0"
)

# ==========================================
# Input Schema
# ==========================================

class Customer(BaseModel):
    Age: int
    Income: float
    Education: str
    Marital_Status: str
    Children: int
    Customer_Tenure: int
    Recency: int
    NumDealsPurchases: int
    NumWebVisitsMonth: int
    Previous_Purchase: int
    Total_Spending: float
    TotalAcceptedCmp: int = 0      # NEW: prior campaign acceptance count
    Complain: int = 0               # NEW: complaint history (0/1)


# ==========================================
# Home Endpoint
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Customer Purchase Prediction API is Running!",
        "version": "2.0",
        "threshold": round(decision_threshold, 3),
        "models_loaded": True
    }


# ==========================================
# Health Check Endpoint
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": round(decision_threshold, 3)
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(customer: Customer):

    input_data = pd.DataFrame(
        [{
            "Age": customer.Age,
            "Income": customer.Income,
            "Education": customer.Education,
            "Marital_Status": customer.Marital_Status,
            "Children": customer.Children,
            "Customer_Tenure": customer.Customer_Tenure,
            "Recency": customer.Recency,
            "NumDealsPurchases": customer.NumDealsPurchases,
            "NumWebVisitsMonth": customer.NumWebVisitsMonth,
            "Previous_Purchase": customer.Previous_Purchase,
            "Total_Spending": customer.Total_Spending,
            "TotalAcceptedCmp": customer.TotalAcceptedCmp,   # NEW
            "Complain": customer.Complain,                    # NEW
        }]
    )

    # Get probability of the positive class (purchase)
    probability = model.predict_proba(input_data)[0]
    prob_likely = float(probability[1])
    prob_not_likely = float(probability[0])

    # Apply tuned decision threshold instead of default 0.5
    prediction = 1 if prob_likely >= decision_threshold else 0

    confidence = prob_likely if prediction == 1 else prob_not_likely

    return {
        "prediction": "Likely Buyer" if prediction == 1 else "Not Likely Buyer",
        "confidence": round(confidence, 3),
        "probability_not_likely": round(prob_not_likely, 3),
        "probability_likely": round(prob_likely, 3),
        "threshold_used": round(decision_threshold, 3),
        "features_used": list(input_data.columns)
    }


# ==========================================
# Batch Prediction Endpoint
# ==========================================

class BatchCustomer(BaseModel):
    customers: list[Customer]

@app.post("/predict_batch")
def predict_batch(batch: BatchCustomer):
    """Predict for multiple customers at once."""

    input_data = pd.DataFrame([
        {
            "Age": c.Age,
            "Income": c.Income,
            "Education": c.Education,
            "Marital_Status": c.Marital_Status,
            "Children": c.Children,
            "Customer_Tenure": c.Customer_Tenure,
            "Recency": c.Recency,
            "NumDealsPurchases": c.NumDealsPurchases,
            "NumWebVisitsMonth": c.NumWebVisitsMonth,
            "Previous_Purchase": c.Previous_Purchase,
            "Total_Spending": c.Total_Spending,
            "TotalAcceptedCmp": c.TotalAcceptedCmp,
            "Complain": c.Complain,
        }
        for c in batch.customers
    ])

    probabilities = model.predict_proba(input_data)
    prob_likely = probabilities[:, 1]

    predictions = (prob_likely >= decision_threshold).astype(int)
    confidences = np.where(predictions == 1, prob_likely, 1 - prob_likely)

    return {
        "predictions": [
            {
                "prediction": "Likely Buyer" if pred == 1 else "Not Likely Buyer",
                "confidence": round(float(conf), 3),
                "probability_likely": round(float(pl), 3),
                "probability_not_likely": round(float(1 - pl), 3),
            }
            for pred, conf, pl in zip(predictions, confidences, prob_likely)
        ],
        "threshold_used": round(decision_threshold, 3),
        "total_customers": len(batch.customers)
    }