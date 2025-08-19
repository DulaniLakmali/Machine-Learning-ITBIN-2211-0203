Iris Classification API

A FastAPI-based machine learning API for predicting Iris flower species using a trained model (`model.pkl`).  
The API takes flower measurements (sepal length, sepal width, petal length, petal width) and returns the predicted species along with a confidence score.

Features
- Health check endpoint to verify API status  
- Prediction endpoint for Iris species classification  
- Model information endpoint to inspect model details  
- Built with FastAPI,pydantic, and scikit-learn 
- Supports JSON input/output  

Requirements
Make sure you have the following installed:

- Python 3.8+
- FastAPI
- Uvicorn
- scikit-learn
- joblib
- numpy

Install dependencies:
bash
pip install fastapi uvicorn scikit-learn joblib numpy
