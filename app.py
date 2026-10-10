from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Enable browser access
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# Load the saved model
bundle = joblib.load("gym_brain.pkl")
model = bundle["model"]
expected_cols = bundle["columns"]


class GymInput(BaseModel):
  tier: str  # Metro, Tier-2, or Suburban
  area_sqft: int
  has_pool: int
  has_sauna: int


@app.post("/predict")
def predict(data: GymInput):
  input_df = pd.DataFrame([data.model_dump()])
  encoded = pd.get_dummies(input_df, columns=["tier"])

  # Fill missing category columns with 0
  for col in expected_cols:
    if col not in encoded.columns:
      encoded[col] = 0
  encoded = encoded[expected_cols]

  estimate = model.predict(encoded)[0]
  return {"estimated_monthly_rate": round(float(estimate), 2)}