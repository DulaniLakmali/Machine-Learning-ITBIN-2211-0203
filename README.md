# Iris Classification FastAPI

## Problem
Classify iris flowers (setosa, versicolor, virginica) from 4 numeric features.

## Model
- Dataset: scikit-learn iris
- Algorithm: RandomForestClassifier
- Test accuracy: printed by `model_dev.py`

## Endpoints
- `GET /` – health check
- `POST /predict` – prediction
- `GET /model-info` – metadata

## Run
```bash
pip install -r requirements.txt
python model_dev.py
uvicorn main:app --reload
