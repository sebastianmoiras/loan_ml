from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from db.connection import get_connection
from db.utils import insert_prediction

app = FastAPI(title="Loan Prediction API")

class LoanData(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: float
    previous_loan_defaults_on_file: str

preprocessor = joblib.load("../model/preprocessor.pkl")
model = joblib.load("../model/best_xgb.pkl")

@app.get("/")
def home():
    return {"message":"Loan Prediction API running"}

@app.post("/predict")
def predict(data: LoanData):
    try:
        data = pd.DataFrame([data.dict()]) 
        input = preprocessor.transform(data) 
        pred_class = model.predict(input)[0]
        proba_value = model.predict_proba(input)[0][pred_class]
        proba = model.predict_proba(input)[0] 

        conn = get_connection()
        if conn:
            insert_prediction(conn, data, pred_class, proba_value)

        return { 
            "status": "success", 
            "prediction": int(pred_class),
            "confidence": float(proba_value),
            "probabilities": proba.tolist()
        } 
    
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
