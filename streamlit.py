import streamlit as st
import requests
import pandas as pd

BASE_URL = st.secrets.get("BASE_URL", "http://127.0.0.1:8000")
st.title("Loan Prediction")
st.write("Input data below: ")

person_age = st.number_input("Age", 20, 100)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education", ["high School", "associate", "bachelor", "master", "doctorate"])
person_income = st.number_input("Income", min_value=0, max_value=10000000, step=1000)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=80)
person_home_ownership = st.selectbox("Home Ownership", ["rent", "own", "mortgage"])
loan_amnt = st.number_input("Loan Amount", min_value=100, max_value=100000, step=500)
loan_intent = st.selectbox("Loan Intent", ["education", "personal", "homeimprovement", "medical", "venture", "debtconsolidation"])
loan_int_rate = st.slider("Interest Rate (%)", 0.0, 20.0, step=0.01, format="%.2f")
loan_percent_income = st.slider("Loan Percent Income", 0.0, 100.0, step=0.01, format="%.2f")
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50, step=1)
credit_score = st.number_input("Credit Score", 300, 1000)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Default on File", ["yes", "no"])

if st.button("Predict"):
    payload = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction successful!")
            st.write(f"Prediction Result: {result['prediction']}")
            st.metric(label="Confidence", value=round(result["probabilities"][result["prediction"]], 4))

            st.subheader("Your Data Input:")
            df_input = pd.DataFrame([payload])
            st.dataframe(df_input)
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
