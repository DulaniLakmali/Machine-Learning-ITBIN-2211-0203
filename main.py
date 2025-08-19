from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model_data = joblib.load("model.pkl")
model = model_data["model"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]

app = FastAPI(title="Iris Classification API",
              description="Predict iris species from flower features")

class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        prediction_idx = model.predict(features)[0]
        proba = float(model.predict_proba(features).max())
        return PredictionOutput(
            prediction=str(target_names[prediction_idx]),
            confidence=proba
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "problem_type": "classification",
        "features": feature_names,
        "classes": list(target_names),
    }
