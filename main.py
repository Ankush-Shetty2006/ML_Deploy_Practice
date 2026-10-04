from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Penguin Species Prediction API", 
              description="An API to predict penguin species based on physical measurements.")

model = joblib.load("model.pkl")

class PenguinData(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float


@app.get("/")
def root():
    return {"message": "API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }

@app.post("/predict")
def predict(data: PenguinData):

    features = [[
        data.bill_length_mm,
        data.bill_depth_mm,
        data.flipper_length_mm,
        data.body_mass_g
    ]]

    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0].max())

    return {"prediction": prediction, "confidence": probability}