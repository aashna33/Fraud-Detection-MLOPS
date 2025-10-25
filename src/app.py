from fastapi import FastAPI
import pandas as pd
import lightgbm as lgb
from pydantic import BaseModel
from typing import Optional
import traceback

# =========================================================
# 🎯 Initialize FastAPI app
# =========================================================
app = FastAPI(
    title="IEEE Fraud Detection API",
    description="Predicts the probability of a transaction being fraudulent using LightGBM.",
    version="1.0"
)

# =========================================================
# 📦 Load trained LightGBM model
# =========================================================
try:
    model = lgb.Booster(model_file="models/lightgbm_model.txt")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# =========================================================
# 🧾 Define input schema (for validation)
# =========================================================
class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: Optional[float] = 0.0
    card1: Optional[float] = 0.0
    card2: Optional[float] = 0.0
    card3: Optional[float] = 0.0
    card5: Optional[float] = 0.0
    addr1: Optional[float] = 0.0
    addr2: Optional[float] = 0.0
    DeviceInfo: Optional[float] = 0.0


# =========================================================
# 🏠 Root endpoint
# =========================================================
@app.get("/")
def root():
    return {"message": "✅ Fraud Detection API is running successfully!"}


# =========================================================
# 🔮 Prediction endpoint
# =========================================================
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([transaction.dict()])

        # --- Align columns with model training features ---
        model_features = model.feature_name()
        for col in model_features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[model_features]

        # --- Make prediction ---
        proba = model.predict(df)[0]
        is_fraud = bool(proba > 0.5)

        return {
            "isFraud": is_fraud,
            "fraud_probability": round(float(proba), 4)
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
