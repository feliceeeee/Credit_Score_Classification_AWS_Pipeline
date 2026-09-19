import json
import os

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(df):
    runtime = get_runtime()
    payload = {"instances": df.values.tolist()}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def main():
    st.title("Credit Score Prediction")
    st.write("Predict customer credit score using the deployed SageMaker model.")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=4)
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=5)
    interest_rate = st.number_input("Interest Rate", min_value=0, max_value=60, value=10)
    num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=50, value=2)
    delay_from_due_date = st.number_input("Delay From Due Date", min_value=0, value=5)
    num_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, max_value=60, value=3)
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, value=5.0)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=50, value=2)
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=100.0, value=30.0)
    amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=500.0)
    total_emi_per_month = st.number_input("Total EMI Per Month", min_value=0.0, value=200.0)
    monthly_balance = st.number_input("Monthly Balance", value=1000.0)
    credit_history_age_months = st.number_input("Credit History Age (Months)", min_value=0, value=120)

    credit_mix = st.selectbox(
        "Credit Mix",
        ["Bad", "Standard", "Good"]
    )

    payment_min_amount = st.selectbox(
        "Payment of Minimum Amount",
        ["No", "Yes"]
    )

    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "High_spent_Small_value_payments",
            "Low_spent_Small_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Large_value_payments"
        ]
    )

    if st.button("Predict Credit Score"):
        input_data = pd.DataFrame([
            {
                "Age": age,
                "Monthly_Inhand_Salary": monthly_inhand_salary,
                "Num_Bank_Accounts": num_bank_accounts,
                "Num_Credit_Card": num_credit_card,
                "Interest_Rate": interest_rate,
                "Num_of_Loan": num_of_loan,
                "Delay_from_due_date": delay_from_due_date,
                "Num_of_Delayed_Payment": num_delayed_payment,
                "Changed_Credit_Limit": changed_credit_limit,
                "Num_Credit_Inquiries": num_credit_inquiries,
                "Outstanding_Debt": outstanding_debt,
                "Credit_Utilization_Ratio": credit_utilization_ratio,
                "Total_EMI_per_month": total_emi_per_month,
                "Amount_invested_monthly": amount_invested_monthly,
                "Monthly_Balance": monthly_balance,
                "Credit_Mix": credit_mix,
                "Payment_of_Min_Amount": payment_min_amount,
                "Payment_Behaviour": payment_behaviour,
                "Credit_History_Age_Months": credit_history_age_months,
                "Annual_Income_log": np.log1p(annual_income)
            }
        ])

        try:
            result = invoke_endpoint(input_data)
            predicted_label = result["prediction"]
            confidence = result["confidence"]
            probability_df = pd.DataFrame(
                {
                    "Class": list(result["probabilities"].keys()),
                    "Probability": list(result["probabilities"].values())
                }
            )

            st.subheader("Prediction Result")

            if predicted_label == "Good":
                st.success(f"Predicted Credit Score: {predicted_label}")
            elif predicted_label == "Standard":
                st.warning(f"Predicted Credit Score: {predicted_label}")
            else:
                st.error(f"Predicted Credit Score: {predicted_label}")

            st.write(f"Confidence: {confidence:.2%}")
            st.subheader("Class Probabilities")
            st.dataframe(probability_df, use_container_width=True)

        except NoCredentialsError:
            st.error("AWS credentials not found.")
        except ClientError as e:
            st.error(f"AWS Error: {e.response['Error']['Message']}")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    main()