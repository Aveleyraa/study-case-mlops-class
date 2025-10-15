# src/app.py
from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI(title="ML Model API", version="1.0")

MODEL_PATH = "model.pkl"

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        model = None
        print("⚠️ Model not found!")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(features: list):
    if model is None:
        return {"error": "Model not loaded"}
    prediction = model.predict(np.array(features).reshape(1, -1))
    return {"prediction": prediction.tolist()}
