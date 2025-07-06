# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define the request body schema
class HousingFeatures(BaseModel):
    features: list[float]  # Expecting 8 values

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_path = "models/model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

@app.get("/")
def read_root():
    return {"message": "ML model API is up!"}

@app.post("/predict")
def predict_price(data: HousingFeatures):
    if len(data.features) != 8:
        raise HTTPException(status_code=400, detail="Expected 8 input features")

    features_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"predicted_price": round(prediction[0], 2)}

