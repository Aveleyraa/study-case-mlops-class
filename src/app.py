# src/app.py
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0")

# Cargar el modelo
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return {"prediction": prediction.tolist()}
