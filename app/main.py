
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "tuned_churn_model.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Load expected feature names
if not TRAIN_DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")

feature_names = pd.read_csv(
    TRAIN_DATA_PATH,
    nrows=0
).columns.tolist()

# Optimized threshold
OPTIMAL_THRESHOLD = 0.5479052186188721

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine Learning API for predicting customer churn",
    version="1.0.0"
)

# Allow React frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Customer input structure
class CustomerData(BaseModel):
    customer: dict


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!",
        "status": "success"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Customer churn prediction
@app.post("/predict")
def predict_churn(data: CustomerData):

    try:
        customer_data = data.customer

        # Check missing features
        missing_features = [
            feature for feature in feature_names
            if feature not in customer_data
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing customer features",
                    "missing_features": missing_features
                }
            )

        # Arrange features in training order
        input_df = pd.DataFrame(
            [[customer_data[feature] for feature in feature_names]],
            columns=feature_names
        )

        # Predict churn probability
        probability = model.predict_proba(input_df)[0][1]

        # Apply optimized threshold
        prediction = (
            1 if probability >= OPTIMAL_THRESHOLD else 0
        )

        return {
            "prediction": (
                "Churn" if prediction == 1 else "No Churn"
            ),
            "churn_probability": round(
                float(probability) * 100, 2
            ),
            "threshold": OPTIMAL_THRESHOLD,
            "status": "success"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
