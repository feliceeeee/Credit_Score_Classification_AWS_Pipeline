import json
import os

import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"

FEATURE_NAMES = [
    "Age",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Credit_History_Age_Months",
    "Annual_Income_log"
]

LABEL_MAPPING = {
    0: "Poor",
    1: "Standard",
    2: "Good"
}


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model_credit.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    df = pd.DataFrame(payload["instances"], columns=FEATURE_NAMES)
    return df


def predict_fn(input_data, model):
    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    predicted_label = LABEL_MAPPING[prediction]

    confidence = float(np.max(probabilities))

    probability_dict = {
        LABEL_MAPPING[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": probability_dict
    }


def output_fn(prediction, accept):
    if accept != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported accept type: {accept}")
    return (json.dumps(prediction), JSON_CONTENT_TYPE)