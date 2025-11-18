import numpy as np
from datetime import datetime

def insert_prediction(conn, data, pred_class, pred_proba):
    pred_class = int(pred_class)
    pred_proba = round(float(np.array(pred_proba).item()), 4)

    record = data.iloc[0].to_dict()

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (
            timestamp, person_age, person_gender, person_education,
            person_income, person_emp_exp, person_home_ownership,
            loan_amnt, loan_intent, loan_int_rate, loan_percent_income,
            cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file,
            pred_class, pred_proba
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        datetime.now(),
        record["person_age"],
        record["person_gender"],
        record["person_education"],
        record["person_income"],
        record["person_emp_exp"],
        record["person_home_ownership"],
        record["loan_amnt"],
        record["loan_intent"],
        record["loan_int_rate"],
        record["loan_percent_income"],
        record["cb_person_cred_hist_length"],
        record["credit_score"],
        record["previous_loan_defaults_on_file"],
        pred_class,
        pred_proba
    ))
    conn.commit()
    cur.close()
